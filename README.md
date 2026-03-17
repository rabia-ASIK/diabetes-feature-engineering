# 🩺 Diabetes Prediction with Feature Engineering

## 📌 Project Overview
This project aims to build a machine learning model that predicts whether a person has diabetes based on medical features.

The main focus is on:
- Data preprocessing
- Feature engineering
- Model performance improvement

---

## 📊 Dataset
The dataset is part of the National Institute of Diabetes and Digestive and Kidney Diseases.

- 768 observations
- 8 numerical features + 1 target
- Target: `Outcome` (1 = diabetic, 0 = non-diabetic)

---

## 🚀 Project Steps

### 1. Exploratory Data Analysis (EDA)
- Data overview
- Variable types
- Distribution analysis
- Target analysis
- Correlation analysis

### 2. Data Preprocessing
- Replaced unrealistic 0 values with NaN
- Filled missing values using median
- Outliers were capped using IQR method

### 3. Feature Engineering
New features created:
- Age categories
- BMI categories
- Glucose categories
- Age + BMI combinations
- Age + Glucose combinations
- Insulin classification
- Interaction features (e.g., Glucose * Insulin)

### 4. Encoding
- Label Encoding
- One-Hot Encoding

### 5. Scaling
- StandardScaler applied to numerical variables

### 6. Modeling
- Random Forest Classifier used

---

## 📈 Model Performance

### 🔹 Base Model
- Accuracy: 0.72
- ROC-AUC: 0.81

### 🔹 Final Model (after Feature Engineering)
- Accuracy: 0.75
- ROC-AUC: 0.83

✅ Feature engineering improved model performance.

---

## 🔥 Feature Importance (Top Variables)
- Glucose
- Age * BMI
- Glucose * Insulin
- BMI
- Age

---
## 📂 Project Structure

```text
diabetes-feature-engineering/
│
├── data/
│   └── diabetes.csv
│
├── src/
│   └── diabetes_pipeline.py
│
├── notebooks/
│   └── diabetes_feature_engineering.ipynb
│
├── README.md

## 📓 Notebook Version

For a detailed step-by-step analysis and explanations:

➡️ notebooks/diabetes_feature_engineering.ipynb
```
## ⚙️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/rabia-ASIK/diabetes-feature-engineering.git
```

---

### 2. Navigate to the project directory

```bash
cd diabetes-feature-engineering
```

---

### 3. (Optional but recommended) Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- Windows:
```bash
venv\Scripts\activate
```

- Mac/Linux:
```bash
source venv/bin/activate
```

---

### 4. Install required dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

### 5. Ensure dataset path

Make sure the dataset is located at:

```text
data/diabetes.csv
```

---

### 6. Run the pipeline

```bash
python src/diabetes_pipeline.py
```

---

### 7. (Optional) Run the notebook version

```bash
jupyter notebook notebooks/diabetes_feature_engineering.ipynb
```

---

## ✅ Expected Output

- Exploratory Data Analysis results
- Correlation heatmap
- Model performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Feature importance visualization

---

## 💡 Tip

If you are using PyCharm, you can directly run:

👉 `src/diabetes_pipeline.py`

from the IDE without using terminal commands.```

## 🎯 Key Takeaways

- Data preprocessing is critical for model performance
- Feature engineering significantly improves results
- Domain knowledge helps create meaningful features
```markdown
## 💡 What I Learned

- Real-world datasets require careful preprocessing
- Feature engineering has a strong impact on model performance
- Understanding the data is more important than choosing complex models

## 👩‍💻 Author
Rabia Aşık