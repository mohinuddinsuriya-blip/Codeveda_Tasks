"""
Level 2 - Task 1: Regression Analysis
Dataset: House Prediction Data Set (Boston Housing)
Internship: Codveda Technologies - Data Analysis Intern
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 120

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
columns = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
    'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

df = pd.read_csv('4__house_Prediction_Data_Set.csv', sep=r'\s+', header=None, names=columns)

print("=" * 60)
print("     LEVEL 2 - TASK 1: REGRESSION ANALYSIS")
print("=" * 60)
print(f"\n📌 Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Target Variable: MEDV (Median House Value in $1000s)")
print(f"\nFirst 5 Rows:")
print(df.head())

# ─────────────────────────────────────────────
# 2. PREPARE FEATURES & TARGET
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("📌 Step 1: Preparing Features and Target Variable")
print("─" * 60)

X = df.drop('MEDV', axis=1)
y = df['MEDV']

print(f"   Features (X): {list(X.columns)}")
print(f"   Target   (y): MEDV")

# ─────────────────────────────────────────────
# 3. SPLIT DATASET
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("📌 Step 2: Splitting Dataset into Train & Test Sets")
print("─" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training Set : {X_train.shape[0]} samples (80%)")
print(f"   Testing Set  : {X_test.shape[0]} samples (20%)")

# ─────────────────────────────────────────────
# 4. FEATURE SCALING
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"\n   ✅ Features scaled using StandardScaler")

# ─────────────────────────────────────────────
# 5. TRAIN LINEAR REGRESSION MODEL
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("📌 Step 3: Training Linear Regression Model")
print("─" * 60)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("   ✅ Model trained successfully!")

# ─────────────────────────────────────────────
# 6. INTERPRET COEFFICIENTS
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("📌 Step 4: Model Coefficients")
print("─" * 60)

coeff_df = pd.DataFrame({
    'Feature'    : X.columns,
    'Coefficient': model.coef_.round(4)
}).sort_values('Coefficient', ascending=False)

print(coeff_df.to_string(index=False))
print(f"\n   Intercept: {model.intercept_:.4f}")

# ─────────────────────────────────────────────
# 7. PREDICTIONS & EVALUATION
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("📌 Step 5: Model Evaluation")
print("─" * 60)

y_pred = model.predict(X_test_scaled)

r2   = r2_score(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)

print(f"\n   📊 R-Squared (R²)         : {r2:.4f}  ({r2*100:.2f}% variance explained)")
print(f"   📊 Mean Squared Error (MSE): {mse:.4f}")
print(f"   📊 Root MSE (RMSE)         : {rmse:.4f}")
print(f"   📊 Mean Absolute Error(MAE): {mae:.4f}")

# ─────────────────────────────────────────────
# 8. VISUALIZATION 1 — Actual vs Predicted
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Linear Regression — House Price Prediction', fontsize=16, fontweight='bold')

# Actual vs Predicted
axes[0].scatter(y_test, y_pred, color='#2196F3', alpha=0.7, edgecolors='white', s=60)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             color='#E91E63', lw=2, linestyle='--', label='Perfect Prediction')
axes[0].set_title(f'Actual vs Predicted\n(R² = {r2:.4f})', fontsize=13)
axes[0].set_xlabel('Actual MEDV ($1000s)')
axes[0].set_ylabel('Predicted MEDV ($1000s)')
axes[0].legend()

# Residuals Plot
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, color='#4CAF50', alpha=0.7, edgecolors='white', s=60)
axes[1].axhline(y=0, color='#E91E63', lw=2, linestyle='--')
axes[1].set_title('Residuals Plot', fontsize=13)
axes[1].set_xlabel('Predicted MEDV ($1000s)')
axes[1].set_ylabel('Residuals')

plt.tight_layout()
plt.savefig('regression_actual_vs_predicted.png', bbox_inches='tight')
plt.show()
print("\n   ✅ Saved: regression_actual_vs_predicted.png")

# ─────────────────────────────────────────────
# 9. VISUALIZATION 2 — Feature Coefficients
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 6))
colors = ['#4CAF50' if c > 0 else '#E91E63' for c in coeff_df['Coefficient']]
bars = plt.barh(coeff_df['Feature'], coeff_df['Coefficient'], color=colors, edgecolor='white')
plt.axvline(x=0, color='black', lw=1.5, linestyle='--')
plt.title('Feature Coefficients — Linear Regression', fontsize=15, fontweight='bold')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('regression_coefficients.png', bbox_inches='tight')
plt.show()
print("   ✅ Saved: regression_coefficients.png")

# ─────────────────────────────────────────────
# 10. VISUALIZATION 3 — Residuals Distribution
# ─────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=30, color='#2196F3', edgecolor='white', alpha=0.85)
plt.axvline(x=0, color='#E91E63', lw=2, linestyle='--', label='Zero Residual')
plt.title('Residuals Distribution', fontsize=15, fontweight='bold')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('regression_residuals_dist.png', bbox_inches='tight')
plt.show()
print("   ✅ Saved: regression_residuals_dist.png")

print("\n" + "=" * 60)
print("✅ Regression Analysis Completed!")
print(f"   Best Result: R² = {r2:.4f} ({r2*100:.2f}% accuracy)")
print("=" * 60)
