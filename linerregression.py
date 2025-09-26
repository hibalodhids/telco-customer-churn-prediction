from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  roc_auc_score,confusion_matrix, classification_report

## Understand Data
df = pd.read_csv(r"D:\TelcoCustomerChurn.csv" , encoding="latin1")
print(df['Churn'].value_counts(normalize=True))
##Visualization
## 1. Churn Distribution
# plt.figure(figsize=(6,4))
# sns.countplot(x='Churn', data=df)
# plt.title("Churn Distribution")
# plt.xlabel("Churn (0=No, 1=Yes)")
# plt.ylabel("Count")
# plt.show()
## 2.Churn by contract type
# sns.countplot(x='Contract', hue='Churn', data=df)
# plt.title("Churn by Contract Type")
# plt.show()
## Churn by monthlyCharges
# sns.boxplot(x='Churn',y='MonthlyCharges',data=df)
# plt.title("Churn by MonthlyCharges")
# plt.show()

## Data Processing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.drop('customerID', axis=1)
df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
##Handling missing Data
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
## Encode categorical Variable
df_encoded = pd.get_dummies(df,drop_first=True)
##Scaling numerical features
scaler = StandardScaler()
df_encoded[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(
    df_encoded[['tenure','MonthlyCharges','TotalCharges']]
)

##Model building
##Train and Test
X = df_encoded.drop('Churn', axis=1)   # Features
y = df_encoded['Churn'] 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
##Trian model1
log_model=LogisticRegression(max_iter=1000)
log_model.fit(X_train , y_train)
##train model 2
rf_model = RandomForestClassifier(n_estimators = 100 , random_state=42)
rf_model.fit(X_train , y_train)

##model evaluation
y_pred = log_model.predict(X_test)
y_proba =log_model.predict_proba(X_test)[:,1]
##Metrics
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Precision:", precision_score(y_test,y_pred))
print("Recall:" , recall_score(y_test,y_pred))
print("F1 Score:" , f1_score(y_test,y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
print("Confusion Matrix:\n", cm)
print("Classification report:",classification_report(y_test,y_pred))

#ratio of new employees whos churn
df['EmployeeType'] = df['tenure'].apply(lambda x: 'New' if x < 12 else 'Experienced')
churn_ratio = df.groupby('EmployeeType')['Churn'].mean() * 100
print("\nChurn Ratio (%):")
print(churn_ratio)

#leftover employees
leftover = df[df['Churn'] == 0]
print("\nNumber of Employees still with Company:", len(leftover))
print("Percentage of Employees retained:", round(len(leftover)/len(df)*100, 2), "%")

##insights and recommendation
#Feature Importance
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 3 Churn Factors:")
print(importances.head(3))


print("\n Insights:")
print("1. Month-to-month contracts → highest churn.")
print("2. Low tenure → new customers leave more.")
print("3. High monthly charges → more likely to churn.")

print("\n Recommendations:")
print("1. Offer discounts/loyalty rewards for long-term contracts.")
print("2. Create onboarding programs for new customers.")
print("3. Provide bundles or discounts to high-bill customers.")


