"""
Task 2: Exploratory Data Analysis (EDA)
Dataset: Iris Dataset
Internship: Codveda Technologies - Data Analysis Intern
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams['figure.dpi'] = 120

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
df = pd.read_csv('1__iris.csv')

print("=" * 55)
print("       TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 55)

print("\n📌 Step 1: Dataset Loaded Successfully")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 Rows:")
print(df.head())

# ─────────────────────────────────────────────
# 2. BASIC INFO
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("📌 Step 2: Dataset Info & Data Types")
print("─" * 55)
print(df.info())
print("\nMissing Values:", df.isnull().sum().sum())
print("Duplicates    :", df.duplicated().sum())

# ─────────────────────────────────────────────
# 3. SUMMARY STATISTICS
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("📌 Step 3: Summary Statistics")
print("─" * 55)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

stats = pd.DataFrame({
    'Mean'   : df[numeric_cols].mean(),
    'Median' : df[numeric_cols].median(),
    'Mode'   : df[numeric_cols].mode().iloc[0],
    'Std Dev': df[numeric_cols].std(),
    'Min'    : df[numeric_cols].min(),
    'Max'    : df[numeric_cols].max(),
}).round(4)

print(stats)

print("\nSpecies Distribution:")
print(df['species'].value_counts())

# ─────────────────────────────────────────────
# 4. HISTOGRAMS — Distribution of Features
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("📌 Step 4: Visualizing Data Distributions")
print("─" * 55)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Iris Dataset — Feature Distributions', fontsize=16, fontweight='bold')

colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
for ax, col, color in zip(axes.flatten(), numeric_cols, colors):
    ax.hist(df[col], bins=20, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(col.replace('_', ' ').title(), fontsize=13)
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('eda_histograms.png', bbox_inches='tight')
plt.show()
print("   ✅ Histogram saved as 'eda_histograms.png'")

# ─────────────────────────────────────────────
# 5. BOXPLOTS — Outliers & Spread by Species
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Iris Dataset — Boxplots by Species', fontsize=16, fontweight='bold')

for ax, col in zip(axes.flatten(), numeric_cols):
    sns.boxplot(data=df, x='species', y=col, ax=ax, palette='Set2')
    ax.set_title(col.replace('_', ' ').title(), fontsize=13)
    ax.set_xlabel('Species')

plt.tight_layout()
plt.savefig('eda_boxplots.png', bbox_inches='tight')
plt.show()
print("   ✅ Boxplot saved as 'eda_boxplots.png'")

# ─────────────────────────────────────────────
# 6. SCATTER PLOTS — Feature Relationships
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Iris Dataset — Scatter Plots', fontsize=16, fontweight='bold')

species_colors = {'setosa': '#E91E63', 'versicolor': '#2196F3', 'virginica': '#4CAF50'}

for species, color in species_colors.items():
    subset = df[df['species'] == species]
    axes[0].scatter(subset['sepal_length'], subset['sepal_width'],
                    label=species, color=color, alpha=0.75, edgecolors='white', s=60)
    axes[1].scatter(subset['petal_length'], subset['petal_width'],
                    label=species, color=color, alpha=0.75, edgecolors='white', s=60)

axes[0].set_title('Sepal Length vs Sepal Width', fontsize=13)
axes[0].set_xlabel('Sepal Length (cm)')
axes[0].set_ylabel('Sepal Width (cm)')
axes[0].legend()

axes[1].set_title('Petal Length vs Petal Width', fontsize=13)
axes[1].set_xlabel('Petal Length (cm)')
axes[1].set_ylabel('Petal Width (cm)')
axes[1].legend()

plt.tight_layout()
plt.savefig('eda_scatter.png', bbox_inches='tight')
plt.show()
print("   ✅ Scatter plot saved as 'eda_scatter.png'")

# ─────────────────────────────────────────────
# 7. CORRELATION HEATMAP
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("📌 Step 5: Correlation Between Numerical Features")
print("─" * 55)

corr = df[numeric_cols].corr()
print(corr.round(4))

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
plt.title('Iris Dataset — Correlation Heatmap', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_correlation.png', bbox_inches='tight')
plt.show()
print("   ✅ Heatmap saved as 'eda_correlation.png'")

# ─────────────────────────────────────────────
# 8. PAIRPLOT
# ─────────────────────────────────────────────
pair = sns.pairplot(df, hue='species', palette='Set2',
                    diag_kind='kde', plot_kws={'alpha': 0.7})
pair.fig.suptitle('Iris Dataset — Pairplot', y=1.02, fontsize=15, fontweight='bold')
plt.savefig('eda_pairplot.png', bbox_inches='tight')
plt.show()
print("   ✅ Pairplot saved as 'eda_pairplot.png'")

print("\n" + "=" * 55)
print("✅ EDA Completed Successfully!")
print("   Saved: eda_histograms.png, eda_boxplots.png,")
print("          eda_scatter.png, eda_correlation.png,")
print("          eda_pairplot.png")
print("=" * 55)
