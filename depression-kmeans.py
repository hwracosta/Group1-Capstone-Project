# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = './Depression Dataset.csv'  # Replace with actual path
df = pd.read_csv(file_path)
print(f'Dataset Shape: {df.shape}')
df.head()

# Preprocessing: Encode categorical columns and scale features
X = df.drop(columns=['DEPRESSED'])  # Exclude target column
for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = LabelEncoder().fit_transform(X[column])

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using the elbow method
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
plt.show()

# Fit K-Means with optimal k (assume k=3 based on elbow plot)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataset
df['Cluster'] = clusters

# Analyze cluster centers
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

print("\nDistinctive features for each cluster:")
for i in range(len(cluster_centers)):
    print(f"\nCluster {i}:")
    sorted_features = cluster_centers.iloc[i].sort_values()
    print("Highest values:")
    print(sorted_features[-5:])
    print("\nLowest values:")
    print(sorted_features[:5])

# Compare clusters with actual target variable
if 'DEPRESSED' in df.columns:
    contingency_table = pd.crosstab(df['Cluster'], df['DEPRESSED'])
    print("\nContingency Table (Clusters vs Actual Labels):")
    print(contingency_table)

# Visualize the clusters using the first two features
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', edgecolor='k', alpha=0.7)
plt.colorbar(label='Cluster')
plt.title('K-Means Clustering Visualization')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.show()
