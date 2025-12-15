# Telco Customer Churn Prediction

## 📌 Project Overview

This project focuses on predicting **customer churn** for a telecom company using machine learning techniques. Customer churn refers to customers who stop using a company’s services. Early prediction helps businesses take preventive actions and improve customer retention.

The project uses the **Telco Customer Churn dataset** and applies multiple classification algorithms to compare performance.

---

## 📂 Dataset

* **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
* **Source:** IBM Sample Dataset
* **Target Variable:** `Churn` (Yes / No)

### Key Features

* Customer demographics (gender, senior citizen, partner, dependents)
* Account information (tenure, contract, payment method)
* Services used (internet service, phone service, streaming, etc.)
* Charges (monthly charges, total charges)

---

## ⚙️ Technologies & Libraries Used

### Programming Language

* Python 🐍

### Libraries

* **Data Handling:** pandas, numpy
* **Visualization:** matplotlib, seaborn, plotly, missingno
* **Machine Learning:** scikit-learn
* **Advanced Models:** XGBoost, CatBoost

---

## 🧠 Machine Learning Models Implemented

The following classifiers are used and compared:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Naive Bayes
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Multi-layer Perceptron (MLP)
* AdaBoost
* Gradient Boosting
* Extra Trees Classifier
* XGBoost Classifier
* CatBoost Classifier

---

## 🔄 Workflow

1. **Import Libraries**
2. **Load Dataset**
3. **Exploratory Data Analysis (EDA)**

   * Missing value analysis
   * Data distribution and correlations
4. **Data Preprocessing**

   * Encoding categorical variables
   * Feature scaling using `StandardScaler`
5. **Train-Test Split**
6. **Model Training**
7. **Model Evaluation**

   * Accuracy Score
   * Precision, Recall, F1-score
   * Confusion Matrix
   * ROC Curve
8. **Model Comparison**

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report
* ROC Curve & AUC

---

## ▶️ How to Run the Project

1. Clone the repository or download the files
2. Install required dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost catboost missingno
   ```
3. Open the Jupyter Notebook:

   ```bash
   jupyter notebook code.ipynb
   ```
4. Run all cells sequentially

---

## 📈 Results

* Multiple models are trained and evaluated
* Ensemble models like **Random Forest, XGBoost, and CatBoost** generally provide higher accuracy
* Performance comparison helps select the best model for churn prediction

---

## 🚀 Future Improvements

* Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
* Handling class imbalance
* Feature engineering
* Model deployment using Flask or FastAPI

---

## 👤 Author

* **Name:** Divyanshi Singh
* **Project Type:** Machine Learning / Data Science

---

## 📜 License

This project is for **educational purposes only**.
