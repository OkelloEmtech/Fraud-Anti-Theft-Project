#FastAPI for Custom Fraud Detection.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn
from typing import Dict, Any, List
import numpy as np

#Add libraries for DB connection
import psycopg2
from sqlalchemy import create_engine, text
from datetime import datetime

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "database": "Fruad_DB",
    "user": "postgres", 
    "password": "1234",
    "port": "5432"
}

engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")



app = FastAPI(title="Fraud Detection API", version="1.0")

# Load model
model = joblib.load('model_fraud.pickle')

#validation using pydantic base_model
class CustomTransaction(BaseModel):
    
    Transaction_Type: str
    Time_of_Transaction: float
    Device_Used: str 
    Location: str
    Previous_Fraudulent_Transaction: int 
    Account_Age: int
    Number_of_Transactions_Last_24H: int
    Payment_Method: str 

#root end point
@app.get('/')
def Home():
    return {'Welcome': 'Fraud Prediction'}

#prediction end point
@app.get("/predict")
async def predict_fraud(
    Transaction_ID: str,
    User_ID: int,
    Transaction_Amount: float,
    Transaction_Type: str,
    Time_of_Transaction: float,
    Device_Used: str,
    Location: str,
    Previous_Fraudulent_Transactions: int,
    Account_Age: int,
    Number_of_Transactions_Last_24H: int,
    Payment_Method: str,
):
    try:
        # Convert float → datetime (this will match DataFrame format)
        Time_of_Transaction = pd.to_datetime(Time_of_Transaction, unit='h')
        
        
        data = {
            'Transaction_ID': [Transaction_ID],
            'User_ID': [User_ID],
            'Transaction_Amount': [Transaction_Amount],
            'Transaction_Type': [Transaction_Type],
            'Time_of_Transaction': [Time_of_Transaction],  # datetime64 format
            'Device_Used': [Device_Used],
            'Location': [Location],
            'Previous_Fraudulent_Transactions': [Previous_Fraudulent_Transactions],
            'Account_Age': [Account_Age],
            'Number_of_Transactions_Last_24H': [Number_of_Transactions_Last_24H],
            'Payment_Method': [Payment_Method]
        }
        #create a data frame
        df = pd.DataFrame(data)
        print(f"DataFrame shape: {df.shape}, dtypes: {df.dtypes}")

        #make prediction 
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]
        
        result = {
            "fraudulent": bool(pred),
            "probability": round(float(prob), 2),
            "risk_level": "HIGH" if prob > 0.8 else "MEDIUM" if prob > 0.3 else "LOW",
            "confidence": round(float(max(model.predict_proba(df)[0])), 2)
        }
        #save to DB
        df_result = pd.DataFrame([{
            'transaction_id': Transaction_ID,
            'user_id': User_ID,
            'transaction_amount': Transaction_Amount,
            'prediction': bool(pred),
            'probability': round(float(prob), 4),
            'risk_level': result['risk_level']
        }])
        df_result.to_sql('fraud_predictions', engine, if_exists='append', index=False, method='multi')
        
        return result

    except Exception as e:
        return {"error": str(e)}
        
  
    


          
        

@app.post("/predict_batch")
async def predict_batch(transactions: List[CustomTransaction]):
    try:
        # Convert to DataFrame + EXACT SAME PROCESSING as /predict
        input_data = []
        for t in transactions:
            dt = pd.to_datetime(t.Time_of_Transaction, unit='h')
            input_data.append({
                'Transaction_ID': t.Transaction_ID,
                'User_ID': t.User_ID,
                'Transaction_Amount': t.Transaction_Amount,
                'Transaction_Type': t.Transaction_Type,
                'Time_of_Transaction': dt,  
                'Device_Used': t.Device_Used,
                'Location': t.Location,
                'Previous_Fraudulent_Transactions': t.Previous_Fraudulent_Transaction,
                'Account_Age': t.Account_Age,
                'Number_of_Transactions_Last_24H': t.Number_of_Transactions_Last_24H,
                'Payment_Method': t.Payment_Method
            })
        
        df = pd.DataFrame(input_data)
        
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]
        
        return [{
            "fraudulent": bool(preds),
            "probability": round(float(probs), 2),
            "risk_level": "HIGH" if probs > 0.8 else "MEDIUM" if probs > 0.3 else "LOW",
            "confidence": round(float(max(model.predict_proba(df.iloc[i:i+1])[0])), 2)
        } for i, (p, prob) in enumerate(zip(preds, probs))]
        
    except Exception as e:
        return [{"error": str(e)} for _ in transactions]






@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

