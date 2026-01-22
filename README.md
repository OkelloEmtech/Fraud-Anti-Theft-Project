# fraud_detection_system
Documentation

Fraud Detection System Overview
Fraud detection system identifies suspicious transactions using machine learning models like Random Forest, Logistic Regression, and XGBoost. It processes transaction data through preprocessing, feature engineering, and model predictions to flag potential fraud. The system integrates with FastAPI for deployment.

Key Components
Data Pipeline: Ingests transaction data, cleans it (handling missing values, outliers), and engineers features like transaction amount ratios and time-based patterns.
​

ML Models: Trained on historical Bank data with scikit-learn; evaluates using metrics like precision, recall, AUC_ROC and F1-score for imbalanced fraud data.

API Layer: FastAPI endpoints serve predictions; example: POST /predict with JSON payload returns fraud probability score.

USing streamlit for a simple front end.