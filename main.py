import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from dataframe import df

# Pre processing
df = pd.get_dummies(df, columns=['plan-type'], drop_first=True)

# Map churn values to 0 and 1
df['churn'] = df['churn'].map({'yes': 1, 'no': 0})
df['churn'] = df['churn'].fillna(0)

# Visualization 
plt.figure(figsize=(10, 6))
sns.countplot(x='churn', hue='call_sac', data=df)
plt.title('Relationship between Call SAC and Churn')
plt.show()

# Clusterization
features = ['subscription-time', 'use-service', 'call_sac', 'client-satisfaction', 'complaints'] + \
           [col for col in df.columns if 'plan-type_' in col]

# Remove outliers
df[features] = df[features].fillna(df[features].mean())

X_cluster = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Final clusterization
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualization with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='viridis', s=100)
plt.title('Cluster Visualization with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Classification Model
X = df[features + ['cluster']]
y = df['churn'].astype(int)

# Balance verification
print("\nBalance Verification:")
print(y.value_counts())

# Data division
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# RandomForest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Avaliation
y_pred = rf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Random Forest')
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Prediction function
def predict_churn(sample_data):
    sample_data = sample_data.copy()
    sample_data = pd.get_dummies(sample_data, columns=['plan-type'], drop_first=True)
    required_cols = ['subscription-time', 'use-service', 'call_sac', 
                    'client-satisfaction', 'complaints']
    plan_type_cols = [col for col in df.columns if 'plan-type_' in col]
    
    for col in plan_type_cols:
        if col not in sample_data.columns:
            sample_data[col] = 0
    
    # Select and order columns exactly as in training
    feature_cols = required_cols + plan_type_cols
    sample_data = sample_data[feature_cols]
    
    sample_data = sample_data.fillna(df[feature_cols].mean())
    
    # Scale features
    sample_scaled = scaler.transform(sample_data)
    
    # Predict cluster
    clusters = kmeans.predict(sample_scaled)
    sample_with_cluster = np.hstack([sample_scaled, clusters.reshape(-1, 1)])
    
    # Predict probabilities and classes
    try:
        probabilities = rf.predict_proba(sample_with_cluster)[:, 1]
        predictions = rf.predict(sample_with_cluster)
    except IndexError:
        # Fallback if model only predicts one class
        predictions = rf.predict(sample_with_cluster)
        probabilities = np.where(predictions == 1, 0.99, 0.01)
    
    return probabilities, predictions

sample_data = pd.DataFrame({
    'subscription-time': [12, 24, 6],
    'use-service': [200, 150, 300],
    'call_sac': [2, 0, 5],
    'client-satisfaction': [3, 4, 2],
    'complaints': [1, 0, 3],
    'plan-type': ['premium', 'basic', 'basic']
})

probabilities, predictions = predict_churn(sample_data)

sample_data['churn_probability'] = probabilities
sample_data['churn_prediction'] = predictions
sample_data['churn_prediction'] = sample_data['churn_prediction'].map({1: 'yes', 0: 'no'})

print("\nPrevisões de Churn para Dados de Exemplo:")
print(sample_data[['subscription-time', 'use-service', 'churn_probability', 'churn_prediction']])