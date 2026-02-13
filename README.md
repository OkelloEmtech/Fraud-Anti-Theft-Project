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

Here's the updated documentation with all code examples removed for cleaner readability.

***

# Anti-Money Laundering (AML) Fraud Detection System

## System Overview
This ML-powered system detects suspicious transactions indicative of money laundering using ensemble models (XGBoost, Random Forest, Logistic Regression). It processes real-time transaction data through automated pipelines, flags high-risk activities, and integrates seamlessly with FastAPI for scalable API serving. A Streamlit dashboard provides an intuitive front-end for monitoring predictions and performance metrics.

**Key Goals**:
- Achieve high precision/recall on imbalanced banking data.
- Support regulatory compliance (KYC/AML standards).
- Deploy as microservice for production fintech environments.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Ingestion│───▶│   ML Pipeline    │───▶│   FastAPI API   │
│ (Transactions)  │    │ (Preprocessing)  │    │ (/predict)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                   │                     │
                                   ▼                     ▼
                          ┌─────────────────┐    ┌─────────────────┐
                          │  Model Training │    │  Streamlit UI   │
                          │ (XGBoost etc.)  │    │ (Dashboard)     │
                          └─────────────────┘    └─────────────────┘
```

## Data Pipeline

### Ingestion & Preprocessing
- **Sources**: Historical bank transactions (PostgreSQL/MongoDB), real-time streams (e.g., MTN MoMo API).
- **Cleaning Steps**:
  - Handle missing values (imputation with median/mode).
  - Detect/remove outliers (IQR method).
  - Encode categoricals (one-hot for account types).
- **Feature Engineering** (critical for AML):
  

## Machine Learning Models

### Model Selection
| Model | Strengths | Use Case |
|-------|-----------|----------|
| **XGBoost** | Handles imbalance, feature importance | Primary model (high AUC) |
| **Random Forest** | Robust to outliers, interpretable | Secondary ensemble |
| **Logistic Regression** | Fast baseline, linear separability | Quick scoring |

### Training & Evaluation
- **Imbalance Handling**: SMOTE oversampling + class weights.
- **Metrics** (prioritize recall for fraud):
  | Metric | Target | Rationale |
  |--------|--------|-----------|
  | Precision | >0.85 | Minimize false positives |
  | Recall | >0.90 | Catch most laundering |
  | AUC-ROC | >0.95 | Overall discrimination |
  | F1-Score | >0.87 | Balanced measure |

- **Hyperparameters** (XGBoost): n_estimators=200, learning_rate=0.1, max_depth=4, reg_alpha=0.1 (L1), reg_lambda=1.5 (L2), scale_pos_weight=10, early_stopping_rounds=10.

## FastAPI Deployment

### Core Endpoints
```
POST /predict
- Payload: {"account_id": "123", "amount": 5000, "time": "2026-02-13T12:00", ...}
- Response: {"fraud_prob": 0.92, "risk_level": "HIGH", "features_importance": {...}}

GET /health
- Status: {"status": "healthy", "model_version": "v1.2"}

POST /batch_predict
- Bulk scoring for transaction batches
```

## Streamlit Dashboard

### Features
- Real-time prediction interface.
- Model performance charts (confusion matrix, ROC curve).
- Feature importance visualization.
- Transaction history table with filters.

**Dashboard Layout**:
```
┌─────────────────────────────┐
│ Transaction Predictor │ ROC │
│ [Input Form] [Predict]      │
├─────────────────────────────┤
│ Feature Importance Bar Chart│
├─────────────────────────────┤
│ Recent Predictions Table    │
└─────────────────────────────┘
```

## Deployment & Monitoring

### Docker Setup
Multi-stage Docker build with Python 3.11 slim base image, uvicorn server on port 8000.

### Monitoring
- MLflow for experiment tracking.
- Prometheus for API metrics.
- Retraining Trigger: Daily on new labeled data; alert if AUC drops <0.92.

### Key Dependencies
xgboost==2.1.1, scikit-learn==1.5.1, fastapi==0.115.0, streamlit==1.38.0, uvicorn==0.30.6, mlflow==2.15.1.

## Usage Instructions

1. **Local Dev**: Install dependencies and run uvicorn main:app --reload.
2. **Predict**: POST to http://localhost:8000/predict with JSON payload.
3. **Dashboard**: Launch Streamlit on port 8501.
4. **Production**: Deploy via Docker Compose with Postgres/Redis.

## Future Enhancements
- Integrate blockchain transaction tracing.
- Add explainable AI (SHAP values).
- AutoML for dynamic model selection.
- Support for MTN MoMo Uganda API.

***

