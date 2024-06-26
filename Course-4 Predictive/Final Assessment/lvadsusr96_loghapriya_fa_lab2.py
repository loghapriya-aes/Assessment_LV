# -*- coding: utf-8 -*-
"""LVADSUSR96_Loghapriya_FA_Lab2

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DgHsOxy81EAkMlAP0RhEWq61ICZJ4grt
"""

import pandas as pd
data = pd.read_csv("/content/sample_data/auto-mpg.csv")
data.head()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print(data.isnull().sum())
data.fillna(data.mean(), inplace=True)
print(df.describe())
print(df.info())
print("\nShape of data:",df.shape)

data.drop(['car name'], axis=1, inplace=True)
data.drop(['horsepower'],axis=1, inplace=True)

# Handle outliers using IQR method
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Histogram of numerical features
data.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Pairplot of numerical features
sns.pairplot(data)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

X = data.drop('mpg', axis=1)
y = data['mpg']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

plt.scatter(y_test, y_pred)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Actual vs. Predicted MPG')
plt.show()

coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
coefficients.sort_values(by='Coefficient', ascending=False, inplace=True)
sns.barplot(x='Coefficient', y=coefficients.index, data=coefficients)
plt.title('Feature Importance')
plt.show()