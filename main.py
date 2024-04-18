# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset (replace 'dataset.csv' with your actual dataset file)
data = pd.read_csv('dataset.csv')

# Data preprocessing
# Drop irrelevant columns for clustering
selected_features = ['Age', 'Annual_Income_(k$)', 'Spending_Score']

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[selected_features])

# (EDA)
# Summary statistics
print("Summary Statistics:")
print(data[selected_features].describe())

# Pairplot for visualizing relationships between features
sns.pairplot(data[selected_features])
plt.title('Pairplot of Selected Features')
plt.show()

# Correlation heatmap
correlation_matrix = data[selected_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Customer Segmentation using K-means clustering
# Determine optimal number of clusters using silhouette score
silhouette_scores = []
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    silhouette_scores.append(silhouette_score(scaled_data, cluster_labels))

# Plot silhouette scores
plt.plot(range(2, 6), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()

# Choose optimal number of clusters based on silhouette score
optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal number of clusters: {optimal_num_clusters}")

# Perform K-means clustering with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(data=data, x='Annual_Income_(k$)', y='Spending_Score', hue='Cluster', palette='Set1', legend='full')
plt.title('Customer Segmentation')
plt.show()

# Analysis of each cluster
for cluster in range(optimal_num_clusters):
    cluster_data = data[data['Cluster'] == cluster][selected_features]
    print(f"\nCluster {cluster} Analysis:")
    print(cluster_data.describe())
