import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def perform_clustering(X, n_clusters=3, random_state=42):
    """
    Do clustering using KMeans algorithm.
    1. Standardize the features
    2. Fit KMeans model
    3. Predict clusters
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters, kmeans, scaler

def plot_elbow_method(X_scaled, max_clusters=10):
    """Plot the elbow method to find the optimal number of clusters"""
    wcss = []
    for i in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters+1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

def plot_clusters_pca(X_scaled, clusters):
    """Visualization clusters using PCA"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=100)
    plt.title('Cluster Visualization with PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()