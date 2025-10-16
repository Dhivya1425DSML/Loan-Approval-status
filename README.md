# 🏦 Loan Approval Prediction (ML Project with EDA & Visualizations)

## 📘 Overview
This project predicts whether a loan application will be **approved or not** using multiple Machine Learning algorithms.  
It includes **Exploratory Data Analysis (EDA)**, **data visualization**, **model training**, and **evaluation** using various classifiers.

The dataset used:
> [loan_approval.csv](https://raw.githubusercontent.com/Dhivya1425DSML/Loan-Approval-status/refs/heads/main/loan_approval.csv)

---

## 📊 Features
- Comprehensive **EDA** with visualizations using `matplotlib` and `seaborn`.
- Preprocessing with **scaling**, **encoding**, and **feature transformation**.
- Multiple **ML classifiers** tested and compared:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Gradient Boosting  
  - Support Vector Machine (SVM)  
  - K-Nearest Neighbors (KNN)  
  - Naive Bayes  
- Model evaluation using:
  - Accuracy, Precision, Recall, F1-score  
  - Confusion Matrix  
  - ROC-AUC Curve  
- Hyperparameter tuning with **GridSearchCV**
- Feature importance visualization for tree-based models

---

## 🧰 Technologies Used
| Category | Tools / Libraries |
|-----------|-------------------|
| Language | Python 3.x |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Model Evaluation | accuracy_score, roc_auc_score, classification_report |

---

## 🧠 Project Workflow

### 1️⃣ Load the Dataset
Load and inspect the dataset directly from the GitHub URL.

### 2️⃣ Exploratory Data Analysis (EDA)
Perform descriptive analysis, handle missing values, and visualize key relationships:
- **Count plots** for target variable  
- **Histograms** for numeric features  
- **Correlation heatmap**  
- **Boxplots** comparing features vs approval status  

### 3️⃣ Data Preprocessing
- Drop irrelevant columns (`name`, `city`)
- Encode categorical features using `OneHotEncoder`
- Scale numeric features with `StandardScaler`
- Split data into **train (80%)** and **test (20%)**

### 4️⃣ Model Building & Evaluation
Train multiple classifiers using Scikit-learn Pipelines and evaluate on test data:
```python
classifiers = [
    (LogisticRegression(max_iter=1000), "Logistic Regression"),
    (DecisionTreeClassifier(random_state=42), "Decision Tree"),
    (RandomForestClassifier(random_state=42), "Random Forest"),
    (GradientBoostingClassifier(random_state=42), "Gradient Boosting"),
    (SVC(probability=True, random_state=42), "SVM"),
    (KNeighborsClassifier(), "K-Nearest Neighbors"),
    (GaussianNB(), "Naive Bayes")
]
