"""
Task 1: Data Cleaning and Preprocessing
Dataset: House Prediction Data Set
Internship: Codveda Technologies - Data Analysis Intern
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. LOAD THE DATASET
# ─────────────────────────────────────────────
columns = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
    'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

df = pd.read_csv('4__house_Prediction_Data_Set.csv', sep=r'\s+', header=None, names=columns)

print("=" * 55)
print("       TASK 1: DATA CLEANING & PREPROCESSING")
print("=" * 55)

print("\n📌 Step 1: Dataset Loaded Successfully")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 Rows:")
print(df.head())

# ─────────────────────────────────────────────
# 2. BASIC INFO
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("📌 Step 2: Dataset Info")
print("─" * 55)
print(df.info())

print("\nData Types:")
print(df.dtypes)

# ─────────────────────────────────────────────
# 3. IDENTIFY MISSING VALUES
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("📌 Step 3: Checking for Missing Values")
print("─" * 55)

# Artificially introduce missing values for demonstration
np.random.seed(42)
for col in ['CRIM', 'ZN', 'AGE', 'LSTAT']:
    idx = np.random.choice(df.index, size=10, replace=False)
    df.loc[idx, col] = np.nan

missing = df.isnull().sum()
print("\nMissing Values Per Column:")
print(missing[missing > 0])
print(f"\nTotal Missing Values: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 4. HANDLE MISSING VALUES
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("📌 Step 4: Handling Missing Values")
print("─" * 55)

# Fill numerical missing values with median
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"   ✅ '{col}' filled with median = {median_val:.4f}")

print(f"\nMissing values after imputation: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 5. REMOVE DUPLICATES
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("📌 Step 5: Checking and Removing Duplicates")
print("─" * 55)

# Introduce duplicates for demonstration
df = pd.concat([df, df.sample(5, random_state=1)], ignore_index=True)
print(f"Duplicate rows before removal: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Duplicate rows after removal : {df.duplicated().sum()}")
print(f"Dataset shape after cleaning : {df.shape}")

# ─────────────────────────────────────────────
# 6. STANDARDIZE DATA FORMATS
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("📌 Step 6: Standardizing Data Formats")
print("─" * 55)

# CHAS is binary (0 or 1) — convert to category
df['CHAS'] = df['CHAS'].astype(int).astype('category')
print("   ✅ 'CHAS' converted to categorical (binary: 0/1)")

# RAD is ordinal index — convert to int
df['RAD'] = df['RAD'].astype(int)
print("   ✅ 'RAD' converted to integer")

# Round float columns to 4 decimal places
float_cols = df.select_dtypes(include=[np.float64]).columns
df[float_cols] = df[float_cols].round(4)
print(f"   ✅ Float columns rounded to 4 decimal places: {list(float_cols)}")

# ─────────────────────────────────────────────
# 7. OUTLIER DETECTION (IQR Method)
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("📌 Step 7: Outlier Detection (IQR Method)")
print("─" * 55)

numeric_cols = df.select_dtypes(include=[np.number]).columns
outlier_report = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    count = ((df[col] < lower) | (df[col] > upper)).sum()
    if count > 0:
        outlier_report[col] = count

print("\nColumns with Outliers:")
for col, cnt in outlier_report.items():
    print(f"   ⚠️  {col}: {cnt} outlier(s)")

# ─────────────────────────────────────────────
# 8. FINAL SUMMARY STATISTICS
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("📌 Step 8: Final Summary Statistics")
print("─" * 55)
print(df.describe())

# ─────────────────────────────────────────────
# 9. SAVE CLEANED DATASET
# ─────────────────────────────────────────────
df.to_csv('house_cleaned.csv', index=False)
print("\n" + "=" * 55)
print("✅ Cleaned dataset saved as 'house_cleaned.csv'")
print("=" * 55)
