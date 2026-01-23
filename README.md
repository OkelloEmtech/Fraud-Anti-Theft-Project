# fraud\_detection/ Anti Theft Investigation\_system



## **Documentation**



### Fraud Detection System Overview



Simple, standalone tool for compliance teams to analyze transaction data, detect fraud and money laundering hence generating investigation reports. 

Its very simple, just upload CSV files, get inference on suspicious cases and export evidence.

One simple app and does everything: Analyze transaction, score each transaction(probability + risk level)- store results in postgresSQL, create investigative ready reports



### **What it solves** 

-This solves too many false alerts(False Positive) by using robust Machine Learning algorithm(Random Forest and XGboost).

-NO manual excel and time consuming techniques.

-Ready to file reports and automation.





### **How it works**

-For simple prediction you input values while on batch processing you simply upload a CSV file.

-The ML model scores every transaction.

-Risk levels are provided from **LOW/MEDIUM/HIGH.**



Risk level logic: "risk\_level": "HIGH" if probabilty > 0.8 else "MEDIUM" if probabilty > 0.3 else "LOW"



### **Database Storage (PostgreSQL)**



Table: Fraud Predictions

&nbsp;    columns:

&nbsp;           'transaction\_id': Transaction\_ID,

&nbsp;           'user\_id': User\_ID,

&nbsp;           'transaction\_amount': Transaction\_Amount,

&nbsp;           'prediction': bool(pred),

&nbsp;           'probability': round(float(prob), 4),

&nbsp;           'risk\_level': result\['risk\_level']

&nbsp;           'created\_at': timestamp



### **Technology Stack**

**-**Programing Language(Python)

-ML: Sklearn-Random Forest Classifier (fraud detection)

-Backend : FastAPI

-Database: PostgreSQL

-Frontend: Streamlit



### **API Endpoints**

&nbsp;          {

&nbsp;           'Transaction\_ID': T1

&nbsp;           'User\_ID': 4589,

&nbsp;           'Transaction\_Amount': 3000,

&nbsp;           'Transaction\_Type': , 'mobile'

&nbsp;           'Time\_of\_Transaction': 10,  # datetime64 format

&nbsp;           'Device\_Used': \[Device\_Used],

&nbsp;           'Location': \[Location], 'Boston',

&nbsp;           'Previous\_Fraudulent\_Transactions': 0,

&nbsp;           'Account\_Age': 20,

&nbsp;           'Number\_of\_Transactions\_Last\_24H': 3,

&nbsp;           'Payment\_Method': 'debit'

&nbsp;           }

&nbsp;            

&nbsp;{

&nbsp; "fraud": true,

&nbsp; "probability": 0.87,

&nbsp; "confidence": 0.92,

&nbsp; "risk\_level": "LOW"

}



Batch processing

post /predict\_batch -saves all results to postgres automatically.



### **Model Performance**

Accuracy: 95%

F1-Score: 97% (balanced fraud/non-fraud)

AUC-ROC: 0.58



### **Feature Importance(TOP Features that contributed the most in model output)**



**Total features after preprocessing: 7797**

**Top 5 Most Important Features:**

                                

**1                Transaction\_Amount    0.0795**

**0                           User\_ID    0.0789**

**3                       Account\_Age    0.0755**

**4   Number\_of\_Transactions\_Last\_24H    0.0588**

**2  Previous\_Fraudulent\_Transactions    0.0423**



### **Key Benefits**

-Results Stored for retraining.

-lesser investigation time.

-Fewer False alerts(False Positive)

-Process batch transactions.



CONFUSION MATRIX

<img width="1050" height="700" alt="training_confusion_matrix" src="https://github.com/user-attachments/assets/d285ced1-f078-4306-988c-b56f48f4c04d" />

ui

<img width="1887" height="945" alt="Screenshot 2026-01-23 102031" src="https://github.com/user-attachments/assets/f044d6f2-5dfd-4d06-a568-da1574b1277f" />







