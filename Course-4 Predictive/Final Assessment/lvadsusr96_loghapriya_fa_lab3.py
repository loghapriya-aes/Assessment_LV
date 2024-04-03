# -*- coding: utf-8 -*-
"""LVADSUSR96_Loghapriya_FA_Lab3

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Yll4b3CuAQopVpHNEMF9alq3-ZB7Y7n2
"""

import pandas as pd
df = pd.read_csv("/content/sample_data/seeds.csv")
df

df.columns

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

df.describe()
print(df.shape)
print(df.info())
print(df.describe())

# Histogram of numerical features
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Pairplot of numerical features
sns.pairplot(df, diag_kind='kde')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='summer', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

inertia_values = []
silhouette_scores = []
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))

plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Optimal k value')
plt.xticks(k_values)
plt.show()

plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Curve for Optimal k value')
plt.xticks(k_values)
plt.show()

optimal_k = 7
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(scaled_data)

cluster_labels = kmeans.predict(scaled_data)

silhouette_avg = silhouette_score(scaled_data, cluster_labels)
print("Average silhouette score: ",silhouette_avg)

df['Cluster'] = kmeans.labels_
cluster_profiles = df.groupby('Cluster').mean()
print(cluster_profiles)

