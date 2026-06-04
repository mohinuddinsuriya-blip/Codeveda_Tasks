"""
Level 3 - Task 1: Predictive Modeling (Classification)
Dataset: Churn Dataset (churn-bigml-80 for training, churn-bigml-20 for testing)
Internship: Codveda Technologies - Data Analysis Intern
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_curve, auc)
from sklearn.model_selection import GridSearchCV

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 120

# ─────────────────────────────────────────────
# 1. LOAD DATASETS
# ─────────────────────────────────────────────
train_df = pd.read_csv('churn-bigml-80.csv')
test_df  = pd.read_csv('churn-bigml-20.csv')

print("=" * 65)
print("   LEVEL 3 - TASK 1: PREDICTIVE MODELING (CLASSIFICATION)")
print("=" * 65)
print(f"\n📌 Training Set : {train_df.shape[0]} rows × {train_df.shape[1]} columns")
print(f"   Testing Set  : {test_df.shape[0]} rows × {test_df.shape[1]} columns")
print(f"\nChurn Distribution (Train):")
print(train_df['Churn'].value_counts())

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("📌 Step 1: Preprocessing — Handling Categorical Variables")
print("─" * 65)

def preprocess(df):
    df = df.copy()
    # Drop non-informative columns
    df.drop(columns=['State', 'Area code'], inplace=True)
    # Encode binary categorical columns
    le = LabelEncoder()
    for col in ['International plan', 'Voice mail plan', 'Churn']:
        df[col] = le.fit_transform(df[col])
    return df

train_df = preprocess(train_df)
test_df  = preprocess(test_df)

print("   ✅ Dropped: 'State', 'Area code'")
print("   ✅ Encoded: 'International plan', 'Voice mail plan', 'Churn'")
print(f"\nFeatures after preprocessing: {list(train_df.columns)}")

# Features & Target
X_train = train_df.drop('Churn', axis=1)
y_train = train_df['Churn']
X_test  = test_df.drop('Churn', axis=1)
y_test  = test_df['Churn']

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"\n   ✅ Feature scaling applied (StandardScaler)")
print(f"   Training: {X_train.shape[0]} samples | Testing: {X_test.shape[0]} samples")

# ─────────────────────────────────────────────
# 3. TRAIN MULTIPLE MODELS
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("📌 Step 2: Training Multiple Classification Models")
print("─" * 65)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree'      : DecisionTreeClassifier(random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        'model'    : model,
        'y_pred'   : y_pred,
        'Accuracy' : accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall'   : recall_score(y_test, y_pred),
        'F1-Score' : f1_score(y_test, y_pred),
    }
    print(f"\n   ✅ {name} trained")

# ─────────────────────────────────────────────
# 4. EVALUATE MODELS
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("📌 Step 3: Model Evaluation — Accuracy, Precision, Recall, F1")
print("─" * 65)

metrics_df = pd.DataFrame({
    name: {k: v for k, v in res.items() if k not in ['model', 'y_pred']}
    for name, res in results.items()
}).T.round(4)

print(f"\n{metrics_df.to_string()}")

best_model_name = metrics_df['F1-Score'].idxmax()
print(f"\n   🏆 Best Model: {best_model_name} (F1={metrics_df.loc[best_model_name,'F1-Score']:.4f})")

# ─────────────────────────────────────────────
# 5. VISUALIZATION 1 — Model Comparison
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
metrics_plot = metrics_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
x = np.arange(len(metrics_plot.columns))
width = 0.25
colors = ['#2196F3', '#4CAF50', '#FF9800']

for i, (model_name, row) in enumerate(metrics_plot.iterrows()):
    bars = ax.bar(x + i * width, row.values, width,
                  label=model_name, color=colors[i], alpha=0.85, edgecolor='white')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

ax.set_title('Model Comparison — Classification Metrics', fontsize=15, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_plot.columns)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
ax.legend()
plt.tight_layout()
plt.savefig('classification_model_comparison.png', bbox_inches='tight')
plt.show()
print("\n   ✅ Saved: classification_model_comparison.png")

# ─────────────────────────────────────────────
# 6. VISUALIZATION 2 — Confusion Matrices
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Confusion Matrices — All Models', fontsize=15, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    ax.set_title(f'{name}\nAcc={res["Accuracy"]:.3f}', fontsize=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('classification_confusion_matrix.png', bbox_inches='tight')
plt.show()
print("   ✅ Saved: classification_confusion_matrix.png")

# ─────────────────────────────────────────────
# 7. VISUALIZATION 3 — ROC Curves
# ─────────────────────────────────────────────
plt.figure(figsize=(9, 6))
roc_colors = ['#2196F3', '#4CAF50', '#FF9800']

for (name, res), color in zip(results.items(), roc_colors):
    if hasattr(res['model'], 'predict_proba'):
        y_prob = res['model'].predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
plt.title('ROC Curves — All Models', fontsize=15, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('classification_roc_curves.png', bbox_inches='tight')
plt.show()
print("   ✅ Saved: classification_roc_curves.png")

# ─────────────────────────────────────────────
# 8. HYPERPARAMETER TUNING — Random Forest
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("📌 Step 4: Hyperparameter Tuning (Random Forest — Grid Search)")
print("─" * 65)

param_grid = {
    'n_estimators'     : [50, 100],
    'max_depth'        : [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=0
)
grid_search.fit(X_train_scaled, y_train)

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_scaled)

print(f"\n   Best Parameters : {grid_search.best_params_}")
print(f"   Best CV F1 Score: {grid_search.best_score_:.4f}")
print(f"\n   Tuned Model Performance on Test Set:")
print(f"   Accuracy : {accuracy_score(y_test, y_pred_best):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred_best):.4f}")
print(f"   Recall   : {recall_score(y_test, y_pred_best):.4f}")
print(f"   F1-Score : {f1_score(y_test, y_pred_best):.4f}")

# ─────────────────────────────────────────────
# 9. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
feat_imp = pd.Series(best_rf.feature_importances_,
                     index=X_train.columns).sort_values(ascending=True)

plt.figure(figsize=(10, 7))
colors = ['#E91E63' if v > feat_imp.median() else '#2196F3' for v in feat_imp]
feat_imp.plot(kind='barh', color=colors, edgecolor='white')
plt.title('Feature Importance — Tuned Random Forest', fontsize=15, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('classification_feature_importance.png', bbox_inches='tight')
plt.show()
print("   ✅ Saved: classification_feature_importance.png")

print("\n" + "=" * 65)
print("✅ Predictive Modeling (Classification) Completed!")
print(f"   Best Model  : {best_model_name}")
print(f"   After Tuning: F1 = {f1_score(y_test, y_pred_best):.4f}")
print("=" * 65)
