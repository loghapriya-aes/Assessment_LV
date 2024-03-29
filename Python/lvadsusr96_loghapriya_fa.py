# -*- coding: utf-8 -*-
"""LVADSUSR96-Loghapriya-FA

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TcPaCObLJqxbdNsTnSGaJJNQrr5v8yeo
"""

import warnings
warnings.filterwarnings('ignore')

#1
import pandas as pd

data = pd.read_excel('/content/Walmart_Dataset Python_Final_Assessment.xlsx')
data.info()

#2
missing_values = data.isna().sum()
print(missing_values)
dup = data.duplicated().sum()
print(dup)

#3
mean_data = data.mean()
print('Mean:\n',mean_data)
median_data = data.median()
print('\nMedian:\n',median_data)
mode_data = data.mode()
print('\nMode:\n',mode_data)
std_values = data.std()
print('\nStandard deviation:\n',std_values)
range_values = data['Sales'].max() - data['Sales'].min()
print('\nRange of Sales column:\n', range_values)
variance_values = data.var()
print('\nVariance:\n',variance_values)

#4
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.DataFrame(data)
def barchart():
  sales_categ = df.groupby('Category')['Sales'].sum()
  plt.figure(figsize=(8,4))
  sales_categ.plot(kind='bar', color='skyblue')
  plt.title('Total Sales by Category')
  plt.xlabel('Index')
  plt.ylabel('Sales')
  plt.show()

def piechart():
  prof_categ = df.groupby('Category')['Quantity'].sum()
  plt.figure(figsize=(6,6))
  prof_categ.plot(kind='pie', autopct='%1.1f%%',labels=df['Category'])
  plt.ylabel('')
  plt.title('Total quantity by Category')
  plt.show()

def linechart():
  prof_categ = df.groupby('Category')['Profit'].mean()
  plt.figure(figsize=(6,6))
  prof_categ.plot(kind='line',marker='o')
  plt.ylabel('')
  plt.xticks(rotation=90)
  plt.grid(True)
  plt.title('Average Profit by Category')
  plt.show()

barchart()
print('\n\n')
piechart()
print('\n\n')
linechart()

#5
correlation_matrix = df.corr()
print(correlation_matrix)

#6
import numpy as np
numeric_cols = ['Sales', 'Quantity', 'Profit']
df_zscores = df[numeric_cols].apply(lambda x: np.abs((x - x.mean()) / x.std()))

outliers = df_zscores > 3

outliers_data = df[outliers.any(axis=1)]

print("Outliers:")
print(outliers_data)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numeric_cols])
plt.title('Boxplot of Sales, Quantity, and Profit')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#7.Trend analysis(i)
data['Year'] = data['Order Date'].dt.year
sales_profit = data.groupby('Year')[['Sales', 'Profit']].sum()

plt.figure(figsize=(12, 6))
plt.plot(sales_profit.index, sales_profit['Sales'], label='Sales', marker='o')
plt.plot(sales_profit.index, sales_profit['Profit'], label='Profit', marker='o')
plt.title('Sales and Profit Trends Over the Years')
plt.xlabel('Year')
plt.ylabel('Amount')
plt.legend()
plt.grid(True)
plt.show()

"""The above plot shows that the sales has increases rapidly with year but the profit has a gradual grpwth."""

# 7.(ii)
data['Year'] = data['Order Date'].dt.year

sales_by_year_category = data.groupby(['Year', 'Category'])['Sales'].sum().unstack()

plt.figure(figsize=(10, 6))
for category in sales_by_year_category.columns:
    plt.plot(sales_by_year_category.index, sales_by_year_category[category], label=category,marker='o')

plt.title('Sales by Category Growth Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.legend(bbox_to_anchor=(1,1),loc='upper left')
plt.grid(True)
plt.xticks(sales_by_year_category.index)
plt.tight_layout()
plt.show()

# 7 Customer Analysis (i)

import pandas as pd

orders_count = df.groupby('EmailID')['Order ID'].count().reset_index()
orders_count.columns = ['EmailID', 'orders_count']
total_sales = df.groupby('EmailID')['Sales'].sum().reset_index()
total_sales.columns = ['EmailID', 'total_sales']
customer_data = pd.merge(orders_count, total_sales, on='EmailID')
top_customers = customer_data.sort_values(by=['orders_count', 'total_sales'], ascending=False).head(5)

print("Top 5 customers based on number of orders and total sales:")
print(top_customers)

"""7 i. It is not proportional that customers with high order count has high sales."""

#7 (ii)
df['Order Date'] = pd.to_datetime(df['Order Date'])
df.sort_values(by=['EmailID', 'Order Date'], inplace=True)
df['time_diff'] = df.groupby('EmailID')['Order Date'].diff()
average_time_between_orders = df.groupby('EmailID')['time_diff'].mean().reset_index()

average_time_between_orders.columns = ['EmailID', 'average_time_between_orders']
print("Average time between orders for each customer:")
print(average_time_between_orders)

top_repeated_users = average_time_between_orders.sort_values(by='average_time_between_orders', ascending=True).head(5)
print("Top five repeated users based on the number of orders:")
print(top_repeated_users)

#7 Comprehensive Analysis
#average Time between order and delivery
df['TimeBetweenOrderAndDelivery'] = df['Ship Date'] - df['Order Date']
average_time_between_order_and_delivery = df.groupby('Category')['TimeBetweenOrderAndDelivery'].mean()
print(average_time_between_order_and_delivery)

df['TimeBetweenOrderAndDelivery'] = df['Ship Date'] - df['Order Date']
average_time_between_order_and_delivery = df.groupby('EmailID')['TimeBetweenOrderAndDelivery'].mean()
print(average_time_between_order_and_delivery.mean())

#i)

import plotly.express as px
fig = px.scatter_geo(df, locations="Geography", locationmode="USA-states",
                      hover_name="Product Name", size="Sales", color="Sales",
                      scope="usa", title="Geographical Distribution")
fig.show()


#ii)

customer_order_amounts = df.groupby('EmailID')['Sales'].sum().reset_index()

top_10_percent = int(len(customer_order_amounts) * 0.1)
high_value_customers = customer_order_amounts.nlargest(top_10_percent, 'Sales')
print("High value Customers based on purchase Value:")
print(high_value_customers)

customer_order_amounts = df.groupby('EmailID')['Quantity'].sum().reset_index()

top_10_percent = int(len(customer_order_amounts) * 0.1)
high_value_customers = customer_order_amounts.nlargest(top_10_percent, 'Quantity')
print("High value Customers based on purchase Quantity:")
print(high_value_customers)

df.sort_values(by=['EmailID', 'Order Date'], inplace=True)
df['TimeBetweenOrders'] = df.groupby('EmailID')['Order Date'].diff()
average_time_between_orders = df.groupby('EmailID')['TimeBetweenOrders'].mean()
print("High value Customers based on purchase Frequency:")
print(average_time_between_orders.nsmallest(top_10_percent))

for index, customer in high_value_customers.iterrows():
  pass

"""### iii) High value customers can be identified by their purchasing quantity, purchase frequency and pruchase amount .
 These customers can be given additional promotions and offers to enhance customer loyalty and they are more likely to recommend wallmart to other potential customers"""

"""**Comprehensive Analysis**

i. Strategies to optimize the supply chain based on insights from sales velocity and order fulfillment data include:

1. Examine past sales velocity information to precisely predict future demand. This lessens stockouts and excess inventory by bringing production and inventory levels into line with actual demand.


2. Order fulfilment data can be used to improve inventory management procedures. By using a just-in-time strategy, carrying expenses and storage space can be reduced while maintaining product availability.


3. Gain valuable insights from sales velocity data to pinpoint important suppliers and build trusting connections with them. Better price, quicker lead times, and more supply chain dependability may result from this.


4. Examine order fulfilment information to enhance delivery itineraries and routes. This can lower the cost of transportation, accelerate delivery, and raise client satisfaction levels all around.

5. Place inventory strategically closer to places with strong demand by using sales velocity data. This speeds up deliveries and lowers transportation expenses.


ii. Factors contributing to the geographic distribution of sales include demographics, economic factors, cultural preferences, competition, and distribution channels. Analyzing these factors can help inform targeted marketing strategies:

1. Recognise the differences in the populations of various areas to adjust marketing messaging and product offerings. Products that are popular with young professionals in cities, for instance, might not appeal to retirees from rural areas.

2. Take into account variables such regional income disparities, employment rates, and disposable income. Adapt pricing and marketing tactics to the state of the local economy.

3. Be aware of how tastes, preferences, and buying habits vary among cultures. Tailor advertising strategies such that they appeal to the regional norms and values.

4. To find market gaps and opportunities, examine the competition environment in each region. Create marketing plans that focus on underrepresented markets and set your products apart from the competition.

5. Adjust distribution channels according to the sales distribution's regional dispersion. Invest in the media outlets that each region's target clients may be reached through most successfully.


iii. Patterns and predictors of high-value customers can include factors such as purchase frequency, average order value, lifetime value, and engagement level. Leveraging this information can enhance customer loyalty and acquisition strategies:

1. Divide up your consumer base according to how valuable each is to the company. Give high-value clients priority when it comes to customised marketing campaigns and exclusive loyalty schemes.

2. Personalisation: Based on unique interests and behaviour, employ data analytics to tailor the consumer experience. Customise messaging, promotions, and product suggestions to appeal to high-value clients.

3. Loyalty programmes: Put in place loyalty programmes that give valuable clients rewards for their continuous business. To encourage recurring business and cultivate loyalty, provide unique access to events, exclusive benefits, or discounts.

4. Customer service: Provide exceptional customer service to high-value customers, resolving issues promptly and going above and beyond to meet their needs. Positive interactions can enhance customer satisfaction and loyalty.

5. Referral programs: Encourage high-value customers to refer their friends and family by offering incentives such as discounts or rewards. Word-of-mouth referrals from satisfied customers can be a powerful driver of acquisition.
"""