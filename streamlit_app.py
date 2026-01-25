import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Fraud Detector", 
    page_icon="🚨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #ff4444; font-weight: bold;}
    .metric-container {padding: 1rem; border-radius: 10px;}
    .stMetric > label {font-size: 1.2rem !important;}
</style>
""", unsafe_allow_html=True)

# Configuration
API_URL = "http://localhost:8000"

# Title and info
st.markdown("# 🚨 **Fraud Detection System**")
st.markdown("**Anti-Theft/Fraud Investigation System ** | *Demo Version*")

# API Health Check
st.sidebar.title("🔌 **API Connection**")
api_url_input = st.sidebar.text_input("FastAPI URL", value=API_URL, help="http://localhost:8000")
test_btn = st.sidebar.button("Test Connection", use_container_width=True)

if test_btn:
    try:
        resp = requests.get(f"{api_url_input}/health", timeout=5)
        if resp.status_code == 200:
            st.sidebar.success("✅ **API Connected!**")
            st.sidebar.json(resp.json())
        else:
            st.sidebar.error(f"❌ **HTTP {resp.status_code}**")
    except Exception as e:
        st.sidebar.error(f"❌ **Connection Failed**: {str(e)}")

# Single Transaction Prediction
st.header("🎯 **Single Transaction Analysis**")

# Input form
with st.form("single_prediction", clear_on_submit=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("**Transaction Details**")
        transaction_id = st.text_input("Transaction ID")
        user_id = st.number_input("User ID", min_value=1)
        transaction_amount = st.number_input("Amount", min_value=0.0)
        transaction_type = st.selectbox("Transaction Type", 
                                      ["ATM Withdrawal", "Bill Payment", "POS Payment", "Online Purchase", "Bank Transfer"])
    
    with col2:
    
        time_of_transaction = st.slider("Time (24h format)", 0.0, 24.0, 14.0)
        device = st.selectbox("Device", ["Mobile", "Tablet", "Desktop", "POS Terminal"])
        location = st.text_input("Location")
        prev_fraud = st.slider("Previous Frauds", 0, 5, 0)
        account_age = st.number_input("Account Age (days)")
        tx_24h = st.number_input("Transactions (24h)", 0, 50, 3)
        payment_method = st.selectbox("Payment Method", 
                                    ["Debit Card", "Credit Card", "Mobile Money", "Bank Transfer"])
    
    predict_btn = st.form_submit_button("🚨 **PREDICT FRAUD**", use_container_width=True, type="primary")

# Process single prediction
if predict_btn and transaction_id:
    try:
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
            "Payment_Method": payment_method
        }
        
        resp = requests.get(f"{api_url_input}/predict", params=params, timeout=10)
        result = resp.json()
        
        if "error" in result:
            st.error(f"❌ **Prediction Error**: {result['error']}")
        else:
            # Results layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                prob_pct = result.get('fraud_probability', 0) * 100
                st.metric("Fraud Probability", f"{prob_pct:.1f}%")
            
            with col2:
                risk = result.get('risk_level', 'LOW')
                risk_color = "🔴 HIGH" if risk == "HIGH" else "🟡 MEDIUM" if risk == "MEDIUM" else "🟢 LOW"
                st.metric("Risk Level", risk_color)
            
            with col3:
                status = "🚨 **FRAUD DETECTED**" if result.get('fraudulent', False) else "✅ **SAFE**"
                st.metric("Status", status)
            
            # Summary
            st.success(f"**{result.get('message', 'Analysis Complete')}**")
            st.json(result)
            
            # Risk breakdown table
            risk_factors = {
                "High Amount": transaction_amount > 1000000,
                "Frequent Tx": tx_24h > 20,
                "New Account": account_age < 30,
                "Past Fraud": prev_fraud > 0,
                "Unknown Location": "Unknown" in location
            }
            
            st.subheader("📊 **Risk Factor Analysis**")
            risk_df = pd.DataFrame(list(risk_factors.items()), columns=["Factor", "Risky"])
            st.dataframe(risk_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ **API Error**: {str(e)}")
        st.info("💡 **Start FastAPI**: `uvicorn main:app --port 8000 --reload`")

# Batch Analysis
st.header("📊 **Batch Analysis**")

col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("📁 Upload CSV File", type="csv", help="Must match API field names")
with col2:
    st.markdown("""
    **Required CSV columns:**
    - Transaction_ID
    - User_ID  
    - Transaction_Amount
    - Transaction_Type
    - Time_of_Transaction
    - Device_Used
    - Location
    - Previous_Fraudulent_Transactions
    - Account_Age
    - Number_of_Transactions_Last_24H
    - Payment_Method
    """)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Preview
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.metric("Total Transactions", len(df))
    
    if st.button("🔥 **Run Batch Prediction**", type="primary", use_container_width=True):
        try:
            resp = requests.post(f"{api_url_input}/predict_batch", json=df.to_dict('records'), timeout=30)
            results = resp.json()
            
            # Add predictions to dataframe
            df['fraud_pred'] = [r.get('fraudulent', False) for r in results]
            df['probability'] = [r.get('fraud_probability', 0) for r in results]
            df['risk_level'] = [r.get('risk_level', 'LOW') for r in results]
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            fraud_rate = df['fraud_pred'].mean()
            high_risk_rate = (df['risk_level'] == 'HIGH').mean()
            
            col1.metric("📈 Fraud Rate", f"{fraud_rate:.1%}")
            col2.metric("🔴 High Risk", f"{high_risk_rate:.1%}")
            col3.metric("💰 Avg Amount", f"UGX {df['Transaction_Amount'].mean():,.0f}")
            
            # Visualization
            fig = px.scatter(
                df, 
                x='Transaction_Amount', 
                y='probability',
                color='fraud_pred',
                size='Number_of_Transactions_Last_24H',
                hover_data=['Transaction_ID', 'Location'],
                title="🔍 Fraud Risk Scatter Plot",
                color_discrete_map={True: '#ff4444', False: '#2ecc71'},
                labels={'Transaction_Amount': 'Amount (UGX)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("📋 **Prediction Results**")
            display_cols = ['Transaction_ID', 'Transaction_Amount', 'probability', 'risk_level', 'fraud_pred']
            st.dataframe(df[display_cols], use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Results CSV",
                data=csv,
                file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ **Batch Error**: {str(e)}")
            st.info("💡 Check: API running, CSV columns match, file size reasonable")

# Footer
st.markdown("---")
st.markdown("""
**Instructions:**
1. Start FastAPI: `uvicorn main:app --port 8000 --reload`
2. Test single prediction first
3. Upload CSV for batch analysis
4. Download results for reporting

**Made for Fraud-Anti-Theft-Project** 🚀
""")
           
