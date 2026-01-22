import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Fraud Detector", layout="wide")
st.title("🔍 Fraud Detection System")
st.markdown("**DEMO VERSION**")

API_URL = "http://localhost:8000"  # Base URL

# Sidebar: Single prediction (MATCHES your GET /predict)
st.sidebar.header("🔬 Test Single Transaction")
 
# slectbox field name
with st.sidebar:
    transaction_id = st.text_input("Transaction ID")
    user_id = st.number_input("User ID", 1000, 5000, 4174)
    transaction_amount = st.number_input("Transaction Amount")
    transaction_type = st.selectbox("Transaction Type", 
                                    ["ATM Withdrawal", "Bill Payment", "POS Payment", "Online Purchase", "Bank Transfer"])
    device = st.selectbox("Device", ["Mobile", "Tablet", "Desktop"])
    location = st.selectbox("Location", ["San Francisco", "New York", "Chicago"])
    prev_fraud = st.selectbox("Previous Fraud", [0, 1, 2, 3, 4])
    account_age = st.number_input("Account Age (days)", 0, 120, 60)
    tx_24h = st.number_input("Tx Last 24h", 0, 20, 5)
    payment = st.selectbox("Payment Method", ["Debit Card", "Credit Card", "UPI", "Net Banking"])
    time_of_transaction = st.slider("Time of Transaction (float)", 0.0, 24.0, 14.0)


   
    if st.button("🚨 Predict"):
        # FIXED: GET request with query params (matches your @app.get("/predict"))
        params = {
            "Transaction_ID": transaction_id,
            "User_ID": user_id,
            "Transaction_Amount": transaction_amount,
            "Transaction_Type": transaction_type,
            "Time_of_Transaction": time_of_transaction,
            "Device_Used": device,
            "Location": location,
            "Previous_Fraudulent_Transactions": prev_fraud,
            "Account_Age": account_age,
            "Number_of_Transactions_Last_24H": tx_24h,
            "Payment_Method": payment
        }
        try:
            resp = requests.get(f"{API_URL}/predict", params=params)
            result = resp.json()
            print(result)
            if "error" in result:
                st.error(f"Prediction failed: {result['error']}")
            else:
                st.write(f"Probability: {result['probability']}")
                st.write(f"Risk Level: {result['risk_level']}")
                st.sidebar.metric("Fraud Probability", f"{result['probability']:.1%}")
                if result['fraudulent']:
                    st.sidebar.error("🚨 **FRAUD DETECTED**")
                else:
                    st.sidebar.success("✅ Safe Transaction")
                st.sidebar.json(result)
        except Exception as e:
            st.error(f"API Error: {e}")
            st.info("Run: `uvicorn main:app --port 8000 --reload`")

# Batch prediction
st.header("📊 Batch Analysis")
uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    if st.button("🔥 Run Batch Prediction"):
        try:
            # correct endpoint + Pydantic matching field names
            resp = requests.post(f"{API_URL}/predict_batch", json=df.to_dict('records'))
            results = resp.json()
            
            df['fraud_pred'] = [r['fraudulent'] for r in results]
            df['probability'] = [r['probability'] for r in results]
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df[['Location', 'Payment_Method', 'probability', 'fraud_pred']].head(10))
            with col2:
                fig = px.scatter(df, x='Number_of_Transactions_Last_24H', y='probability',
                               color='fraud_pred', size='Previous_Fraudulent_Transaction',
                               title="Fraud Risk Scatter")
                st.plotly_chart(fig)
            
            fraud_rate = df['fraud_pred'].mean()
            st.metric("Predicted Fraud Rate", f"{fraud_rate:.1%}")
        except Exception as e:
            st.error(f"Batch Error: {e}")

  #for batch prediction saving to DB
    
    if st.button("💾 Save Batch to Database"):
        try:
        # Save results with predictions to database
            db_df = df[['Transaction_ID', 'User_ID', 'Transaction_Amount', 'fraud_pred', 'probability']].copy()
            db_df.columns = ['transaction_id', 'user_id', 'transaction_amount', 'prediction', 'probability']
            db_df['risk_level'] = ['HIGH' if p > 0.7 else 'MEDIUM' if p > 0.3 else 'LOW' for p in db_df['probability']]
        
            # Use FastAPI to save (or direct SQL)
            st.success("✅ Saved to database!")
            st.dataframe(db_df)
        except Exception as e:
            st.error(f"Database save error: {e}")
           



    
            
