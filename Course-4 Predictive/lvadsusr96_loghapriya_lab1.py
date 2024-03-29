# -*- coding: utf-8 -*-
"""LVADSUSR96_Loghapriya_Lab1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12X6w4UFoooSYN1r2dxr56HpTq_OR-8W7
"""

import pandas as pd
data = pd.read_csv("/content/expenses.csv")
df = pd.DataFrame(data)
# print(df)
missing_values = df.isnull().sum()
print(missing_values)
data = data.dropna()
data = data.drop_duplicates()

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['age', 'bmi', 'smoker', 'charges']])
plt.title('Boxplot for Outlier Detection')
plt.show()

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
cleaned_df = data[~outliers]
print(cleaned_df)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.get_dummies(cleaned_df, columns=['sex', 'smoker', 'region'], drop_first=True)
# df
X = df.drop(columns=['charges'])
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r_squared = r2_score(y_test, predictions)
rmse = np.sqrt(mse)

print("Mean Squared Error (MSE):", mse)
print("R-squared:", r_squared)
print("Root Mean Squared Error (RMSE):", rmse)

age = float(input("Enter age: "))
bmi = float(input("Enter BMI: "))
children = int(input("Enter the number of dependents/children: "))
sex = int(input("Is the insured individual male? (1=Yes, 0=No)): "))
smoker = int(input("Is the insured individual a smoker? (1=Yes, 0=No): "))
region_northwest = int(input("Is the insured individual from the Northwest region? (1=Yes, 0=No)): "))
region_southwest = int(input("Is the insured individual from the Southwest region? (1=Yes, 0=No)): "))
region_southeast = int(input("Is the insured individual from the Southeast region? (1=Yes, 0=No)): "))

input_data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'sex_male': [sex],
    'smoker_yes': [smoker],
    'region_northwest': [region_northwest],
    'region_southeast': [region_southeast],
    'region_southwest': [region_southwest]
})

predicted_charges = model.predict(input_data)
print("Predicted Insurance Charges:", predicted_charges[0])