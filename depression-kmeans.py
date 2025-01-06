# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     sync: true
# ---

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from kneed import KneeLocator

# ## 1. Load the Dataset
print("Step 1: Loading the dataset...")
file_path = '.'  # Replace with actual path
df = pd.read_csv(file_path)
print(f"Dataset loaded successfully! Shape: {df.shape}")
df.head()

# ## 2. Preprocess the Data
print("\nStep 2: Preprocessing the dataset...")
# Exclude target column and encode categorical features
X = df.drop(columns=['DEPRESSED'])
for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = LabelEncoder().fit_transform(X[column])

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data preprocessing complete. Features are now encoded and scaled.")

# ## 3. Determine Optimal Number of Clusters
print("\nStep 3: Determining the optimal number of clusters using the Elbow Method...")
inertias = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertias, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid()
plt.show()

# Use KneeLocator to find the "elbow"
knee = KneeLocator(K, inertias, curve='convex', direction='decreasing')
optimal_k = knee.knee
print(f"Optimal number of clusters detected: k={optimal_k}")

# ## 4. Perform K-Means Clustering
print("\nStep 4: Performing K-Means clustering...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters
print(f"K-Means clustering complete. Clusters assigned for k={optimal_k}.")

# ## 5. Evaluate Clustering with Silhouette Score
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"\nStep 5: Evaluating clustering performance...")
print(f"Silhouette Score: {silhouette_avg:.2f}")
if silhouette_avg > 0.5:
    print("Clusters are well-separated and cohesive.")
elif silhouette_avg > 0.25:
    print("Clusters are reasonably separated but could overlap.")
else:
    print("Clusters are poorly separated; consider revising the clustering approach.")

# ## 6. Analyze Cluster Characteristics
print("\nStep 6: Analyzing distinctive features for each cluster...")
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

for i in range(len(cluster_centers)):
    print(f"\nCluster {i}:")
    sorted_features = cluster_centers.iloc[i].sort_values()
    print("Highest values:")
    print(sorted_features[-5:])
    print("\nLowest values:")
    print(sorted_features[:5])

# ## 7. Compare Clusters with Actual Labels
if 'DEPRESSED' in df.columns:
    print("\nStep 7: Comparing clusters with actual labels...")
    contingency_table = pd.crosstab(df['Cluster'], df['DEPRESSED'])
    print("Contingency Table (Clusters vs Actual Labels):")
    print(contingency_table)

    # Visualize contingency table as heatmap
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues')
    plt.title('Clusters vs Actual Labels')
    plt.ylabel('Cluster')
    plt.xlabel('Depressed (Actual)')
    plt.show()

# ## 8. Visualize Clusters with PCA
print("\nStep 8: Visualizing clusters in 2D using PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', edgecolor='k', alpha=0.7)
plt.colorbar(label='Cluster')
plt.title('K-Means Clustering Visualization (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()

print("\nAll steps completed! The clustering analysis is now ready.")
