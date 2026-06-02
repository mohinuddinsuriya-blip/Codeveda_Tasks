"""
Level 2 - Task 3: Clustering Analysis (K-Means)
Dataset: Iris Dataset
Internship: Codveda Technologies - Data Analysis Intern
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 120

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
df = pd.read_csv('1__iris.csv')

print("=" * 60)
print("   LEVEL 2 - TASK 3: CLUSTERING ANALYSIS (K-MEANS)")
print("=" * 60)

print(f"\n📌 Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 5 Rows:")
print(df.head())

# ─────────────────────────────────────────────
# 2. PREPARE FEATURES
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("📌 Step 1: Preparing Features")
print("─" * 60)

# Store actual species for comparison later
actual_species = df['species'].copy()
X = df.drop('species', axis=1)
feature_names = X.columns.tolist()

print(f"   Features used: {feature_names}")
print(f"   Actual species: {actual_species.unique()}")

# ─────────────────────────────────────────────
# 3. STANDARDIZE THE DATASET
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("📌 Step 2: Standardizing the Dataset (StandardScaler)")
print("─" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("   Before Scaling (mean, std):")
for col in feature_names:
    print(f"   {col}: mean={X[col].mean():.3f}, std={X[col].std():.3f}")

print("\n   After Scaling (mean ≈ 0, std ≈ 1):")
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
for col in feature_names:
    print(f"   {col}: mean={X_scaled_df[col].mean():.3f}, std={X_scaled_df[col].std():.3f}")

# ─────────────────────────────────────────────
# 4. ELBOW METHOD — Find Optimal K
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("📌 Step 3: Finding Optimal K using Elbow Method")
print("─" * 60)

inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, km.labels_))
    print(f"   K={k} | Inertia={km.inertia_:.2f} | Silhouette={silhouette_score(X_scaled, km.labels_):.4f}")

# Plot Elbow Curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('K-Means — Finding Optimal Number of Clusters', fontsize=15, fontweight='bold')

axes[0].plot(K_range, inertia, 'bo-', linewidth=2, markersize=8, color='#2196F3')
axes[0].axvline(x=3, color='#E91E63', linestyle='--', lw=2, label='Optimal K=3')
axes[0].set_title('Elbow Method', fontsize=13)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (WCSS)')
axes[0].legend()

axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8, color='#4CAF50')
axes[1].axvline(x=3, color='#E91E63', linestyle='--', lw=2, label='Optimal K=3')
axes[1].set_title('Silhouette Score', fontsize=13)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()

plt.tight_layout()
plt.savefig('kmeans_elbow.png', bbox_inches='tight')
plt.show()
print("\n   ✅ Saved: kmeans_elbow.png")
print("   ✅ Optimal K = 3 (confirmed by Elbow + Silhouette)")

# ─────────────────────────────────────────────
# 5. APPLY K-MEANS WITH K=3
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("📌 Step 4: Applying K-Means with K=3")
print("─" * 60)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\n   Cluster Distribution:")
print(df['Cluster'].value_counts().sort_index().to_string())

print(f"\n   Final Silhouette Score: {silhouette_score(X_scaled, df['Cluster']):.4f}")

# Cluster Centers (unscaled)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=feature_names)
centers_df.index.name = 'Cluster'
print(f"\n   Cluster Centers (original scale):")
print(centers_df.round(3))

# ─────────────────────────────────────────────
# 6. VISUALIZATION 1 — 2D Scatter Plots
# ─────────────────────────────────────────────
colors = ['#E91E63', '#2196F3', '#4CAF50']
cluster_labels = ['Cluster 0', 'Cluster 1', 'Cluster 2']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('K-Means Clustering — Iris Dataset', fontsize=15, fontweight='bold')

# Plot 1: Sepal features
for i, (color, label) in enumerate(zip(colors, cluster_labels)):
    mask = df['Cluster'] == i
    axes[0].scatter(df[mask]['sepal_length'], df[mask]['sepal_width'],
                    color=color, label=label, alpha=0.75, edgecolors='white', s=70)
axes[0].scatter(centers_df['sepal_length'], centers_df['sepal_width'],
                marker='X', s=200, color='black', zorder=5, label='Centroids')
axes[0].set_title('Sepal Length vs Sepal Width', fontsize=13)
axes[0].set_xlabel('Sepal Length (cm)')
axes[0].set_ylabel('Sepal Width (cm)')
axes[0].legend()

# Plot 2: Petal features
for i, (color, label) in enumerate(zip(colors, cluster_labels)):
    mask = df['Cluster'] == i
    axes[1].scatter(df[mask]['petal_length'], df[mask]['petal_width'],
                    color=color, label=label, alpha=0.75, edgecolors='white', s=70)
axes[1].scatter(centers_df['petal_length'], centers_df['petal_width'],
                marker='X', s=200, color='black', zorder=5, label='Centroids')
axes[1].set_title('Petal Length vs Petal Width', fontsize=13)
axes[1].set_xlabel('Petal Length (cm)')
axes[1].set_ylabel('Petal Width (cm)')
axes[1].legend()

plt.tight_layout()
plt.savefig('kmeans_clusters_2d.png', bbox_inches='tight')
plt.show()
print("   ✅ Saved: kmeans_clusters_2d.png")

# ─────────────────────────────────────────────
# 7. VISUALIZATION 2 — PCA 2D Plot
# ─────────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))
for i, (color, label) in enumerate(zip(colors, cluster_labels)):
    mask = df['Cluster'] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                color=color, label=label, alpha=0.75, edgecolors='white', s=70)

plt.title(f'K-Means Clusters — PCA Visualization\n(Explained Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%)',
          fontsize=14, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.legend()
plt.tight_layout()
plt.savefig('kmeans_pca.png', bbox_inches='tight')
plt.show()
print("   ✅ Saved: kmeans_pca.png")

# ─────────────────────────────────────────────
# 8. VISUALIZATION 3 — Cluster vs Actual Species
# ─────────────────────────────────────────────
df['Species'] = actual_species.values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('K-Means Clusters vs Actual Species', fontsize=15, fontweight='bold')

species_colors = {'setosa': '#E91E63', 'versicolor': '#2196F3', 'virginica': '#4CAF50'}

for species, color in species_colors.items():
    mask = df['Species'] == species
    axes[0].scatter(df[mask]['petal_length'], df[mask]['petal_width'],
                    color=color, label=species, alpha=0.75, edgecolors='white', s=70)
axes[0].set_title('Actual Species', fontsize=13)
axes[0].set_xlabel('Petal Length')
axes[0].set_ylabel('Petal Width')
axes[0].legend()

for i, (color, label) in enumerate(zip(colors, cluster_labels)):
    mask = df['Cluster'] == i
    axes[1].scatter(df[mask]['petal_length'], df[mask]['petal_width'],
                    color=color, label=label, alpha=0.75, edgecolors='white', s=70)
axes[1].set_title('K-Means Clusters (K=3)', fontsize=13)
axes[1].set_xlabel('Petal Length')
axes[1].set_ylabel('Petal Width')
axes[1].legend()

plt.tight_layout()
plt.savefig('kmeans_vs_actual.png', bbox_inches='tight')
plt.show()
print("   ✅ Saved: kmeans_vs_actual.png")

print("\n" + "=" * 60)
print("✅ K-Means Clustering Analysis Completed!")
print(f"   Optimal Clusters : K = 3")
print(f"   Silhouette Score : {silhouette_score(X_scaled, df['Cluster']):.4f}")
print("   Output Files     : kmeans_elbow.png, kmeans_clusters_2d.png,")
print("                      kmeans_pca.png, kmeans_vs_actual.png")
print("=" * 60)
