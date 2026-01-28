from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn
from typing import List
from datetime import datetime
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import traceback

db_url = "postgresql://postgres:1234@localhost:5432/Fruad_DB"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API", 
    version="2.0"
)

#  CREATE BASE FOR ORM MODELS
Base = declarative_base()

#  DEFINE DATABASE MODEL
class PredictionLog(Base):
    __tablename__ = 'fraud_predictions_256'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_id = Column(String, unique=True, index=True)
    fraudulent = Column(Boolean)
    fraud_probability = Column(Float)
    risk_level = Column(String)
    confidence = Column(Float)
    message = Column(String)
    timestamp = Column(DateTime, default=datetime.now)

# Database setup
engine = create_engine(db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ✅ CREATE TABLES IF THEY DON'T EXIST
Base.metadata.create_all(bind=engine)

def get_db_session():
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Don't close here, let the caller handle it

# Model loading
try:
    model = joblib.load('model_fraud.pickle')
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"Model load failed: {e}")
    model = None

class FraudPredictionRequest(BaseModel):
    Transaction_ID: str = Field(..., min_length=1, description="Unique transaction ID")
    User_ID: int = Field(..., gt=0, description="User identifier")
    Transaction_Amount: float = Field(..., gt=0, description="Transaction amount")
    Transaction_Type: str = Field(..., description="Transaction type")
    Time_of_Transaction: float = Field(..., description="Unix timestamp in hours")
    Device_Used: str = Field(..., description="Device used")
    Location: str = Field(..., description="Transaction location")
    Previous_Fraudulent_Transactions: int = Field(0, ge=0, description="Previous fraud count")
    Account_Age: int = Field(0, ge=0, description="Account age in days")
    Number_of_Transactions_Last_24H: int = Field(0, ge=0, description="Recent transactions")
    Payment_Method: str = Field(..., description="Payment method")

@app.get("/")
async def root():
    return {
        "message": "Fraud Detection API v2.0",
        "status": "healthy",
        "model_loaded": model is not None,
        "endpoints": ["/predict (GET)", "/predict_batch (POST)", "/health"]
    }

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
    Payment_Method: str
):
    """Single prediction"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check model_fraud.pickle")
    
    db = get_db_session()  # ✅ Open session early
    
    try:
        # Convert timestamp to datetime
        transaction_time = pd.to_datetime(Time_of_Transaction, unit='h')
        
        # Create prediction DataFrame
        data = {
            'Transaction_ID': [Transaction_ID],
            'User_ID': [User_ID],
            'Transaction_Amount': [Transaction_Amount],
            'Transaction_Type': [Transaction_Type],
            'Time_of_Transaction': [transaction_time],
            'Device_Used': [Device_Used],
            'Location': [Location],
            'Previous_Fraudulent_Transactions': [Previous_Fraudulent_Transactions],
            'Account_Age': [Account_Age],
            'Number_of_Transactions_Last_24H': [Number_of_Transactions_Last_24H],
            'Payment_Method': [Payment_Method]
        }
        
        df = pd.DataFrame(data)
        logger.info(f"🔍 Predicting for transaction {Transaction_ID}")
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        # Risk classification
        risk_level = "HIGH" if probability > 0.8 else "MEDIUM" if probability > 0.3 else "LOW"
        
        # ✅ CREATE SQLAlchemy MODEL INSTANCE (not dictionary)
        prediction_log = PredictionLog(
            transaction_id=Transaction_ID,
            fraudulent=bool(prediction),
            fraud_probability=round(float(probability), 4),
            risk_level=risk_level,
            confidence=round(float(max(model.predict_proba(df)[0])), 4),
            message="🚨 FRAUD DETECTED" if prediction else "✅ SAFE TRANSACTION",
            timestamp=datetime.now()
        )
        
        # ✅ SAVE TO DATABASE
        try:
            db.add(prediction_log)
            db.commit()
            db.refresh(prediction_log)
            logger.info(f"✅ Logged to database: {Transaction_ID}")
        except Exception as e:
            db.rollback()
            logger.error(f"❌ DB error: {e}")
            # Continue even if DB fails - still return prediction
        
        # ✅ RETURN DICTIONARY RESPONSE
        result = {
            "transaction_id": Transaction_ID,
            "fraudulent": bool(prediction),
            "fraud_probability": round(float(probability), 4),
            "risk_level": risk_level,
            "confidence": round(float(max(model.predict_proba(df)[0])), 4),
            "message": "🚨 FRAUD DETECTED" if prediction else "✅ SAFE TRANSACTION",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Prediction complete: {Transaction_ID} -> {risk_level} ({probability:.3f})")
        return result
    
    except Exception as e:
        traceback.print_exc()
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
    
    finally:
        db.close()  # ✅ Fixed: actually call close()

@app.post("/predict_batch")
async def predict_batch(transactions: List[FraudPredictionRequest]):
    """Batch fraud prediction for multiple transactions"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")
    
    db = get_db_session()
    
    try:
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
                'Previous_Fraudulent_Transactions': t.Previous_Fraudulent_Transactions,
                'Account_Age': t.Account_Age,
                'Number_of_Transactions_Last_24H': t.Number_of_Transactions_Last_24H,
                'Payment_Method': t.Payment_Method
            })
        
        df = pd.DataFrame(input_data)
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            risk_level = "HIGH" if prob > 0.8 else "MEDIUM" if prob > 0.3 else "LOW"
            
            # ✅ LOG EACH BATCH PREDICTION
            prediction_log = PredictionLog(
                transaction_id=df.iloc[i]['Transaction_ID'],
                fraudulent=bool(pred),
                fraud_probability=round(float(prob), 4),
                risk_level=risk_level,
                confidence=round(float(prob) if pred else round(float(1-prob), 4), 4),
                message="🚨 FRAUD DETECTED" if pred else "✅ SAFE TRANSACTION",
                timestamp=datetime.now()
            )
            db.add(prediction_log)
            
            results.append({
                "index": i,
                "transaction_id": df.iloc[i]['Transaction_ID'],
                "fraudulent": bool(pred),
                "fraud_probability": round(float(prob), 4),
                "risk_level": risk_level,
                "message": "🚨 FRAUD DETECTED" if pred else "✅ SAFE TRANSACTION"
            })
        
        db.commit()
        logger.info(f"✅ Batch prediction complete: {len(results)} transactions logged")
        return results
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    finally:
        db.close()

@app.get("/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "2.0",
        "endpoints": {
            "single": "GET /predict?...",
            "batch": "POST /predict_batch",
            "docs": "/docs (Swagger UI)"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)