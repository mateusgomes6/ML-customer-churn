import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

from dataframe import df

plt.figure(figsize=(10, 6))
sns.boxplot(x='churn', y='call_sac', data=df)
plt.title('Relação entre Churn e Ligações para o SAC')
plt.show()

features_cluster = ['subscription-time', 'plan-type', 'use-service', 'call_sac', 'client-satisfaction', 'complaints']
X_cluster = df[features_cluster]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=9, init='k-menas++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)