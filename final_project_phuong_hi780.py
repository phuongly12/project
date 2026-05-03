
import pyodbc
import pandas as pd

conn = pyodbc.connect(
    "DRIVER={SQL Server};"
    "SERVER=DESKTOP-O1MFT0I\SQLPHUONG;"
    "DATABASE=HI780;"
    "Trusted_Connection=yes;"
)
df = pd.read_sql("SELECT * FROM cardio", conn)
print(df.shape)
print(df.head())

X = df.drop(columns=["cardio", "id"], errors='ignore')
y = df["cardio"]
X = pd.get_dummies(X, drop_first=True)
train_columns = X.columns
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
y_train = y_train.astype(int)
y_test = y_test.astype(int)


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
y_pred_lr = model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
y_pred_lr = model.predict(X_test)
y_prob_lr = model.predict_proba(X_test)[:, 1]

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf)
    ],
    "Precision": [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_rf)
    ],
    "Recall": [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_rf)
    ],
    "F1-Score": [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_rf)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, y_prob_lr),
        roc_auc_score(y_test, y_prob_rf)
    ]
})

print(results)

import pandas as pd

importance = pd.Series(rf_model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))

new_patient = pd.DataFrame([{
    "age_years": 55,
    "gender": 1,
    "bmi": 35,
    "ap_hi": 160,
    "ap_lo": 95,
    "cholesterol": 2,
    "gluc": 1,
    "smoke": 1,
    "alco": 0,
    "active": 0,
    "bp_category_encoded": 2
}])

new_patient = pd.get_dummies(new_patient)

new_patient = new_patient.reindex(columns=train_columns, fill_value=0)

prediction = rf_model.predict(new_patient)
probability = rf_model.predict_proba(new_patient)[0][1]

print("Risk:", prediction[0])
print("Probability:", probability)
