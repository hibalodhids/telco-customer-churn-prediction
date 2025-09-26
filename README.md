# Telco Customer Churn Prediction

This project predicts customer churn using the **Telco Customer Churn dataset**.  
It includes data preprocessing, exploratory data analysis (EDA), model building, evaluation, and actionable business insights.

---
Name = Hiba Lodhi
Department of Data Science,University of Sindh

## 📌 Project Overview
- Understand customer churn behavior.
- Apply data preprocessing (handle missing values, encoding, scaling).
- Visualize churn distribution and factors affecting churn.
- Train and evaluate machine learning models.
- Identify key churn drivers and provide recommendations.

---

## 📊 Dataset
**File:** `TelcoCustomerChurn.csv`  

**Key Columns:**
- `Churn`: Target variable (1 = churned, 0 = retained)  
- `tenure`, `MonthlyCharges`, `TotalCharges`: Numerical features  
- `Contract`, `InternetService`, `PaymentMethod`, etc.: Categorical features  

---

## ⚙️ Steps in Project
1. **Exploratory Data Analysis (EDA)**  
   - Distribution of churn  
   - Churn by contract type, monthly charges, etc.  

2. **Data Preprocessing**  
   - Handle missing values  
   - Encode categorical variables  
   - Scale numerical features  

3. **Model Building**  
   - Logistic Regression  
   - Random Forest  

4. **Model Evaluation**  
   - Accuracy, Precision, Recall, F1-score  
   - AUC-ROC  
   - Confusion Matrix  

5. **Insights & Recommendations**  
   - Month-to-month contracts have the highest churn.  
   - Low tenure customers are more likely to leave.  
   - High monthly charges increase churn probability.  

---

## 📈 Results
- **Top 3 churn factors (from feature importance):**
  1. Contract type  
  3. Monthly charges  

  2. Tenure  
- **Best Practices to Reduce Churn:**
  - Offer loyalty rewards for long-term contracts.  
  - Improve onboarding for new customers.
 
## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/hibalodhids/telco-customer-churn-prediction.git
   cd telco-customer-churn-prediction
2. Install:
   pip install -r requirements.txt
3.Run :
   
  - Provide discounts/bundles to high-bill customers.  

---
