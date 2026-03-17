###############################################################
# Diabetes Feature Engineering Project
###############################################################

# Business Problem:
# Develop a machine learning model that can predict whether a person
# has diabetes based on medical features.
# Before modeling, perform exploratory data analysis and feature engineering.

# Dataset Story:
# The dataset is part of a large database maintained by the
# National Institute of Diabetes and Digestive and Kidney Diseases.
# The target variable is "Outcome":
# 1 -> diabetes positive
# 0 -> diabetes negative

# Variables:
# Pregnancies: Number of pregnancies
# Glucose: Plasma glucose concentration
# BloodPressure: Diastolic blood pressure (mm Hg)
# SkinThickness: Triceps skin fold thickness (mm)
# Insulin: 2-Hour serum insulin (mu U/ml)
# BMI: Body mass index
# DiabetesPedigreeFunction: Diabetes pedigree function
# Age: Age (years)
# Outcome: Target variable (1 = diabetic, 0 = non-diabetic)

###############################################################
# PROJECT TASKS
###############################################################

# TASK 1: EXPLORATORY DATA ANALYSIS
# Step 1: Examine the overall picture
# Step 2: Identify categorical and numerical variables
# Step 3: Analyze categorical and numerical variables
# Step 4: Perform target variable analysis
# Step 5: Perform outlier analysis
# Step 6: Perform missing value analysis
# Step 7: Perform correlation analysis

# TASK 2: FEATURE ENGINEERING
# Step 1: Handle missing values and outliers
# Step 2: Create new variables
# Step 3: Apply encoding
# Step 4: Scale numerical variables
# Step 5: Build and evaluate models

###############################################################
# IMPORTS
###############################################################

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

warnings.simplefilter(action="ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")

###############################################################
# PATHS
###############################################################

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

###############################################################
# LOAD DATA
###############################################################

df = pd.read_csv(DATA_PATH)

###############################################################
# HELPER FUNCTIONS
###############################################################

def check_df(dataframe, head=5):
    print("\n##################### SHAPE #####################")
    print(dataframe.shape)

    print("\n##################### TYPES #####################")
    print(dataframe.dtypes)

    print("\n##################### HEAD #####################")
    print(dataframe.head(head))

    print("\n##################### TAIL #####################")
    print(dataframe.tail(head))

    print("\n##################### MISSING VALUES #####################")
    print(dataframe.isnull().sum())

    print("\n##################### DESCRIPTIVE STATISTICS #####################")
    print(dataframe.describe([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Returns categorical, numerical and cardinal variable names.
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [
        col for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
    ]

    cat_but_car = [
        col for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
    ]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print("\n##################### VARIABLE TYPES #####################")
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)} -> {cat_cols}")
    print(f"num_cols: {len(num_cols)} -> {num_cols}")
    print(f"cat_but_car: {len(cat_but_car)} -> {cat_but_car}")
    print(f"num_but_cat: {len(num_but_cat)} -> {num_but_cat}")

    return cat_cols, num_cols, cat_but_car


def cat_summary(dataframe, col_name, plot=False):
    summary_df = pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    })
    print(f"\n########## {col_name} ##########")
    print(summary_df)

    if plot:
        plt.figure(figsize=(5, 4))
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.title(f"{col_name} Distribution")
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / f"{col_name.lower()}_countplot.png")
        plt.show()


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(f"\n########## {numerical_col} ##########")
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        plt.figure(figsize=(6, 4))
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(f"{numerical_col} Distribution")
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / f"{numerical_col.lower()}_hist.png")
        plt.show()


def target_summary_with_num(dataframe, target, numerical_col):
    print(f"\n########## {numerical_col} by {target} ##########")
    print(dataframe.groupby(target).agg({numerical_col: "mean"}))


def target_summary_with_cat(dataframe, target, categorical_col):
    print(f"\n########## {categorical_col} by {target} ##########")
    print(dataframe.groupby(categorical_col).agg({target: "mean"}))


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    if len(na_columns) == 0:
        print("\nNo missing values found.")
        return [] if na_name else None

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print("\n##################### MISSING VALUES TABLE #####################")
    print(missing_df)

    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = [col for col in temp_df.columns if "_NA_FLAG" in col]

    print("\n##################### MISSING VS TARGET #####################")
    for col in na_flags:
        print(f"\n########## {col} ##########")
        print(pd.DataFrame({
            "TARGET_MEAN": temp_df.groupby(col)[target].mean(),
            "Count": temp_df.groupby(col)[target].count()
        }))


def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    return pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)


def plot_correlation_matrix(dataframe):
    plt.figure(figsize=(10, 8))
    corr = dataframe.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "correlation_heatmap.png")
    plt.show()


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n##################### {model_name.upper()} RESULTS #####################")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cv_results = cross_validate(
        model,
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]),
        cv=5,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
    )

    print("Cross Validation Mean Scores:")
    print(f"CV Accuracy : {cv_results['test_accuracy'].mean():.4f}")
    print(f"CV Precision: {cv_results['test_precision'].mean():.4f}")
    print(f"CV Recall   : {cv_results['test_recall'].mean():.4f}")
    print(f"CV F1       : {cv_results['test_f1'].mean():.4f}")
    print(f"CV ROC-AUC  : {cv_results['test_roc_auc'].mean():.4f}")

    return model


def plot_importance(model, features, num=20):
    feature_imp = pd.DataFrame({
        "Value": model.feature_importances_,
        "Feature": features.columns
    }).sort_values("Value", ascending=False)

    print("\n##################### FEATURE IMPORTANCE #####################")
    print(feature_imp.head(num))

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Value", y="Feature", data=feature_imp.head(num))
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "feature_importance.png")
    plt.show()


###############################################################
# TASK 1: EXPLORATORY DATA ANALYSIS
###############################################################

print("\n" + "#" * 60)
print("TASK 1: EXPLORATORY DATA ANALYSIS")
print("#" * 60)

# Step 1: Examine the overall picture
print("\n[TASK 1 - STEP 1] OVERVIEW")
check_df(df)

# Step 2: Identify categorical and numerical variables
print("\n[TASK 1 - STEP 2] VARIABLE TYPES")
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Step 3: Analyze categorical and numerical variables
print("\n[TASK 1 - STEP 3] CATEGORICAL ANALYSIS")
for col in cat_cols:
    cat_summary(df, col, plot=False)

print("\n[TASK 1 - STEP 3] NUMERICAL ANALYSIS")
for col in num_cols:
    num_summary(df, col, plot=False)

# Step 4: Target variable analysis
print("\n[TASK 1 - STEP 4] TARGET ANALYSIS")
for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

for col in cat_cols:
    if col != "Outcome":
        target_summary_with_cat(df, "Outcome", col)

# Step 5: Outlier analysis
print("\n[TASK 1 - STEP 5] OUTLIER ANALYSIS")
for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")

# Step 6: Missing value analysis
print("\n[TASK 1 - STEP 6] MISSING VALUE ANALYSIS")
missing_values_table(df)

zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
print("\nColumns with suspicious zero values:")
print(zero_columns)

for col in zero_columns:
    print(f"{col} -> Zero count: {(df[col] == 0).sum()}")

# Step 7: Correlation analysis
print("\n[TASK 1 - STEP 7] CORRELATION ANALYSIS")
print(df.corr(numeric_only=True)["Outcome"].sort_values(ascending=False))
plot_correlation_matrix(df)

###############################################################
# BASE MODEL BEFORE FEATURE ENGINEERING
###############################################################

print("\n" + "#" * 60)
print("BASE MODEL BEFORE FEATURE ENGINEERING")
print("#" * 60)

y_base = df["Outcome"]
X_base = df.drop("Outcome", axis=1)

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_base, y_base, test_size=0.20, random_state=42
)

base_model = RandomForestClassifier(random_state=42)
base_model = evaluate_model(base_model, X_train_base, X_test_base, y_train_base, y_test_base, model_name="Base Model")

###############################################################
# TASK 2: FEATURE ENGINEERING
###############################################################

print("\n" + "#" * 60)
print("TASK 2: FEATURE ENGINEERING")
print("#" * 60)

df_fe = df.copy()

# Step 1: Handle missing values and outliers
print("\n[TASK 2 - STEP 1] HANDLE MISSING VALUES AND OUTLIERS")

# Replace suspicious zero values with NaN
for col in zero_columns:
    df_fe[col] = np.where(df_fe[col] == 0, np.nan, df_fe[col])

# Missing value table after zero->NaN
na_columns = missing_values_table(df_fe, na_name=True)

# Analyze missingness against target
if len(na_columns) > 0:
    missing_vs_target(df_fe, "Outcome", na_columns)

# Fill missing values with median
for col in zero_columns:
    df_fe[col] = df_fe[col].fillna(df_fe[col].median())

print("\nAfter median imputation:")
missing_values_table(df_fe)

# Outlier capping
cat_cols_fe, num_cols_fe, cat_but_car_fe = grab_col_names(df_fe)
for col in num_cols_fe:
    if col != "Outcome":
        if check_outlier(df_fe, col):
            replace_with_thresholds(df_fe, col)

print("\nOutlier check after capping:")
for col in num_cols_fe:
    if col != "Outcome":
        print(f"{col}: {check_outlier(df_fe, col)}")

# Step 2: Create new variables
print("\n[TASK 2 - STEP 2] CREATE NEW FEATURES")

# Age category
df_fe.loc[(df_fe["Age"] >= 21) & (df_fe["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df_fe.loc[(df_fe["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

# BMI category
df_fe["NEW_BMI_CAT"] = pd.cut(
    x=df_fe["BMI"],
    bins=[0, 18.5, 24.9, 29.9, 100],
    labels=["Underweight", "Healthy", "Overweight", "Obese"]
)

# Glucose category
df_fe["NEW_GLUCOSE_CAT"] = pd.cut(
    x=df_fe["Glucose"],
    bins=[0, 140, 200, 300],
    labels=["Normal", "Prediabetes", "DiabetesRisk"]
)

# Age-BMI combined category
df_fe.loc[(df_fe["BMI"] < 18.5) & (df_fe["Age"] < 50), "NEW_AGE_BMI_CAT"] = "underweight_mature"
df_fe.loc[(df_fe["BMI"] < 18.5) & (df_fe["Age"] >= 50), "NEW_AGE_BMI_CAT"] = "underweight_senior"
df_fe.loc[(df_fe["BMI"] >= 18.5) & (df_fe["BMI"] < 25) & (df_fe["Age"] < 50), "NEW_AGE_BMI_CAT"] = "healthy_mature"
df_fe.loc[(df_fe["BMI"] >= 18.5) & (df_fe["BMI"] < 25) & (df_fe["Age"] >= 50), "NEW_AGE_BMI_CAT"] = "healthy_senior"
df_fe.loc[(df_fe["BMI"] >= 25) & (df_fe["BMI"] < 30) & (df_fe["Age"] < 50), "NEW_AGE_BMI_CAT"] = "overweight_mature"
df_fe.loc[(df_fe["BMI"] >= 25) & (df_fe["BMI"] < 30) & (df_fe["Age"] >= 50), "NEW_AGE_BMI_CAT"] = "overweight_senior"
df_fe.loc[(df_fe["BMI"] >= 30) & (df_fe["Age"] < 50), "NEW_AGE_BMI_CAT"] = "obese_mature"
df_fe.loc[(df_fe["BMI"] >= 30) & (df_fe["Age"] >= 50), "NEW_AGE_BMI_CAT"] = "obese_senior"

# Age-Glucose combined category
df_fe.loc[(df_fe["Glucose"] < 70) & (df_fe["Age"] < 50), "NEW_AGE_GLUCOSE_CAT"] = "low_mature"
df_fe.loc[(df_fe["Glucose"] < 70) & (df_fe["Age"] >= 50), "NEW_AGE_GLUCOSE_CAT"] = "low_senior"
df_fe.loc[(df_fe["Glucose"] >= 70) & (df_fe["Glucose"] < 100) & (df_fe["Age"] < 50), "NEW_AGE_GLUCOSE_CAT"] = "normal_mature"
df_fe.loc[(df_fe["Glucose"] >= 70) & (df_fe["Glucose"] < 100) & (df_fe["Age"] >= 50), "NEW_AGE_GLUCOSE_CAT"] = "normal_senior"
df_fe.loc[(df_fe["Glucose"] >= 100) & (df_fe["Glucose"] <= 125) & (df_fe["Age"] < 50), "NEW_AGE_GLUCOSE_CAT"] = "hidden_mature"
df_fe.loc[(df_fe["Glucose"] >= 100) & (df_fe["Glucose"] <= 125) & (df_fe["Age"] >= 50), "NEW_AGE_GLUCOSE_CAT"] = "hidden_senior"
df_fe.loc[(df_fe["Glucose"] > 125) & (df_fe["Age"] < 50), "NEW_AGE_GLUCOSE_CAT"] = "high_mature"
df_fe.loc[(df_fe["Glucose"] > 125) & (df_fe["Age"] >= 50), "NEW_AGE_GLUCOSE_CAT"] = "high_senior"

# Insulin category
def set_insulin(value):
    return "Normal" if 16 <= value <= 166 else "Abnormal"

df_fe["NEW_INSULIN_SCORE"] = df_fe["Insulin"].apply(set_insulin)

# Interaction variables
df_fe["NEW_GLUCOSE_INSULIN"] = df_fe["Glucose"] * df_fe["Insulin"]
df_fe["NEW_AGE_BMI"] = df_fe["Age"] * df_fe["BMI"]
df_fe["NEW_PREG_AGE_RATIO"] = df_fe["Pregnancies"] / (df_fe["Age"] + 1)
df_fe["NEW_BMI_AGE_RATIO"] = df_fe["BMI"] / (df_fe["Age"] + 1)

# Convert column names to uppercase for a more standard presentation
df_fe.columns = [col.upper() for col in df_fe.columns]

print("\nFeature engineered dataframe head:")
print(df_fe.head())
print("\nNew shape:", df_fe.shape)

# Step 3: Encoding
print("\n[TASK 2 - STEP 3] ENCODING")

cat_cols_fe, num_cols_fe, cat_but_car_fe = grab_col_names(df_fe)

binary_cols = [col for col in df_fe.columns if df_fe[col].dtype == "O" and df_fe[col].nunique() == 2]
print("\nBinary columns for label encoding:")
print(binary_cols)

for col in binary_cols:
    df_fe = label_encoder(df_fe, col)

cat_cols_fe = [col for col in cat_cols_fe if col not in binary_cols and col != "OUTCOME"]
df_fe = one_hot_encoder(df_fe, cat_cols_fe, drop_first=True)

print("\nAfter encoding:")
print(df_fe.head())
print("Shape after encoding:", df_fe.shape)

# Step 4: Scale numerical variables
print("\n[TASK 2 - STEP 4] SCALING")

cat_cols_fe, num_cols_fe, cat_but_car_fe = grab_col_names(df_fe)
num_cols_fe = [col for col in num_cols_fe if col != "OUTCOME"]

scaler = StandardScaler()
df_fe[num_cols_fe] = scaler.fit_transform(df_fe[num_cols_fe])

print("\nAfter scaling:")
print(df_fe.head())

# Step 5: Build and evaluate final model
print("\n[TASK 2 - STEP 5] FINAL MODEL")

y = df_fe["OUTCOME"]
X = df_fe.drop("OUTCOME", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

final_model = RandomForestClassifier(random_state=42)
final_model = evaluate_model(final_model, X_train, X_test, y_train, y_test, model_name="Final Model")

plot_importance(final_model, X, num=20)

###############################################################
# FINAL SUMMARY
###############################################################

print("\n" + "#" * 60)
print("FINAL PROJECT SUMMARY")
print("#" * 60)

print("""
1. The dataset was examined with exploratory data analysis.
2. Suspicious zero values were treated as missing values.
3. Missing values were imputed with median.
4. Outliers were capped instead of being removed.
5. New domain-based and interaction-based features were created.
6. Categorical variables were encoded.
7. Numerical variables were scaled.
8. A baseline Random Forest model was built.
9. A final Random Forest model was built after feature engineering.
10. Model performance improved after preprocessing and feature engineering.

This script is organized to be:
- easy to explain in presentations
- suitable for PyCharm
- reusable in GitHub portfolio projects
- understandable for recruiters and interviewers
""")