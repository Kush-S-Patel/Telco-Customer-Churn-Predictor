# 🧠 Telco Customer Churn Prediction Model

A machine learning project designed to **predict customer churn** using data science techniques.  
This repository demonstrates the full ML lifecycle — from data exploration and feature engineering to model training, evaluation, and optional deployment.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Future Work](#-future-work)
- [License](#-license)

---

## 📖 Overview

**Customer churn** is a large focus of businesses, especially subscription-based services.
By knowing exactly what causes churn in existing customers, companies can reduce marketing and acquistion costs, retain valuable customers, and improve customer satisfaction with targeted retention efforts.
This project builds and evaluates several classification models to **identify customers at risk of churning**, using historical data.

---

## 🧱 Project Structure

```
.
├── data/
│   └── data.csv                # Original dataset(s)
├── src/
│   ├── __init__.py
│   └── main.py
├── requirements.txt
└── README.md
```

---

## 🧪 Dataset

The dataset contains anonymized information about telecom customers, including demographics, subscribed services, billing details, and whether they churned.

| Feature              | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `customerID`         | Unique customer identifier                                                  |
| `gender`             | Gender of the customer                                                      |
| `SeniorCitizen`      | Whether the customer is a senior citizen (1 = Yes, 0 = No)                   |
| `Partner`            | Whether the customer has a partner (Yes/No)                                 |
| `Dependents`         | Whether the customer has dependents (Yes/No)                                |
| `tenure`             | Number of months the customer has stayed with the company                   |
| `PhoneService`       | Whether the customer has phone service (Yes/No)                             |
| `MultipleLines`      | Whether the customer has multiple lines                                     |
| `InternetService`    | Type of internet service (DSL, Fiber optic, None)                           |
| `OnlineSecurity`     | Whether the customer has online security service                            |
| `OnlineBackup`       | Whether the customer has online backup service                              |
| `DeviceProtection`   | Whether the customer has device protection                                  |
| `TechSupport`        | Whether the customer has technical support                                  |
| `StreamingTV`        | Whether the customer has streaming TV service                               |
| `StreamingMovies`    | Whether the customer has streaming movies service                           |
| `Contract`          | Type of contract (Month-to-month, One year, Two year)                        |
| `PaperlessBilling`   | Whether the customer uses paperless billing (Yes/No)                        |
| `PaymentMethod`      | Method of payment (e.g., Electronic check, Credit card, Bank transfer)      |
| `MonthlyCharges`     | Monthly amount charged to the customer                                     |
| `TotalCharges`       | Total amount charged to the customer to date                               |
| `Churn`             | Target variable — whether the customer churned (Yes = churned, No = active) |
<img width="600" height="666" alt="Heatmap" src="https://github.com/user-attachments/assets/fc3a55d6-1022-41c8-87ef-e2f1c065e8ca" />


> You can use your own dataset or download the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn) from Kaggle.

---

## 🧠 Approach

1. **Exploratory Data Analysis (EDA)**  
   - Inspected distributions and descriptive statistics for numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`).  
   - Plotted feature correlations with `Churn` to identify the strongest predictors.  
   - Detected and removed outliers using Z-scores.  

2. **Feature Engineering**  
   - Encoded categorical features using `LabelEncoder` and one-hot encoding for model compatibility.  
   - Converted numerical columns to proper numeric types and interpolated missing values.  
   - Cleaned categorical missing values with forward/backward filling.  
   - Removed extreme outliers (`|z| > 3`) to stabilize model training.  

3. **Modeling**  
   - Split the data into train (80%) and test (20%) sets.  
   - Trained multiple algorithms:  
     - **Random Forest** (tuned with deeper trees and leaf constraints)  
     - **Linear Regression** (as a baseline)  
     - **XGBoost Classifier**  
     - **Gradient Boosting Classifier**  
   - Compared performance across models using accuracy and regression metrics where applicable.  

4. **Evaluation**  
   - **Metrics:** Accuracy, Mean Squared Error (for Linear model), R² Score.  
   - Focused on model accuracy for churn classification.  
   - Random Forest and Gradient Boosting achieved the highest performance on the test set.
   
---

## 📈 Model Performance

| Model                      | Train Accuracy | Test Accuracy | MSE   | R²   |
|----------------------------|---------------:|-------------:|------:|-----:|
| **Random Forest**          | 0.8912        | 0.8062      |   —   |  —   |
| **Linear Regression**      | —             | —           | 0.13  | 0.31 |
| **XGBoost Classifier**     | —             | 0.7842      |   —   |  —   |
| **Gradient Boosting**      | 0.8268        | 0.8077      |   —   |  —   |

> Random Forest and Gradient Boosting achieved the highest test accuracy (~80.7%), making them the most effective models for predicting customer churn in this dataset.


## 🛠 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Kush-S-Patel/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
```

## 🚀 Future Work

- [ ] Add model monitoring for production data  
- [ ] Deploy a real-time API endpoint for predictions  
- [ ] Automate retraining pipelines  
- [ ] Experiment with deep learning models (e.g., TabNet, NN)

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for more information.

---

## ✨ Acknowledgments

- [Kaggle Telco Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)  
- scikit-learn, pandas, matplotlib, seaborn  
- XGBoost, Streamlit, Flask

---

Made with ❤️ by [Your Name](https://github.com/your-username)
