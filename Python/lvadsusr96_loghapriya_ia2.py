#1
import numpy as np

def rgb_to_grayscale(rgb_image):
    R = rgb_image[:,:,0]
    G = rgb_image[:,:,1]
    B = rgb_image[:,:,2]

    grayscale_image = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return grayscale_image

rgb_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                      [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
                      [[127, 127, 127], [200, 200, 200], [50, 50, 50]]])

grayscale_image = rgb_to_grayscale(rgb_image)

print("Grayscale Image:\n",grayscale_image)

#2
def normalize(a):
    b = np.zeros(9).reshape(3,3)
    for i in range(len(a)):
        for j in range(len(a[i])):
           b[i][j] = (a[i][j] - a[i].mean())/(a[i].std())

    return b

#3
data = np.array([[[1,2],[2,3]],[[1,2],[3,4]],[[5,8],[9,4]]])
print("3D structure:",data)
flat_data = [a.flatten() for a in data]
ans = np.array(flat_data)

print("Original:",data.shape)
print("Reshaped data format:",ans.shape)

#4
data = np.random.randint(0, 100, size=(3, 5))
print(data)
g1_score = data[:, 0]
g2_score = data[:, -1]
improvement = g2_score - g1_score
for i, j in enumerate(improvement):
    print(f"Athlete {i+1}: Improvement = {j}")

#5
data = np.array([[64,81,52,61,66],
 [70,51, 54, 76, 59],
 [60,-1, 77, 88, 89],
 [54,73, 90, -1, 87],
 [93,70, 86, 79, 63]])

three_sub = data[:, -3:]
valid_scores = np.where(three_sub != -1, three_sub, 0)
sum_valid_scores = np.sum(valid_scores, axis=1)
count_valid_scores = np.count_nonzero(valid_scores, axis=1)
count_valid_scores[count_valid_scores == 0] = 1
average_scores = sum_valid_scores / count_valid_scores
print("Average scores",average_scores)

#6
def apply_adjustment_factors(city_temperatures, adjustment_factors):

    adjusted_factors = adjustment_factors.reshape(1, -1)
    adjusted_temperatures = city_temperatures * adjusted_factors

    return adjusted_temperatures
city_temperatures = np.array([
    [20, 25, 30, 28],
    [15, 18, 22, 20],
    [10, 12, 14, 16]
])
adjustment_factors = np.array([0.95, 0.98, 1.05, 1.02])

adjusted_temperatures = apply_adjustment_factors(city_temperatures, adjustment_factors)

print("Original temperatures:")
print(city_temperatures)
print("\nAdjusted temperatures:")
print(adjusted_temperatures)

#7
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace'],
    'Age': [25, 30, 35, 40, 45, 50, 55],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Miami', 'Boston'],
    'Department': ['HR', 'IT', 'Finance', 'Marketing', 'Sales', 'IT', 'HR']
}

final = [(data['Name'][i], data['City'][i]) for i in range(len(data['Name']))
                 if data['Age'][i] < 45 and data['Department'][i] != 'HR']

for name, city in final:
    print(f"Name: {name}, City: {city}")

#8
import pandas as pd
data = {
    'Product': ['Apples', 'Bananas', 'Cherries', 'Dates', 'Elderberries', 'Flour', 'Grapes'],
    'Category': ['Fruit', 'Fruit', 'Fruit', 'Fruit', 'Fruit', 'Bakery', 'Fruit'],
    'Price': [1.20, 0.50, 3.00, 2.50, 4.00, 1.50, 2.00],
    'Promotion': [True, False, True, True, False, True, False]
}

df = pd.DataFrame(data)
fruit_df = df[df['Category'] == 'Fruit']
average_price = fruit_df['Price'].mean()
potential_candidates = fruit_df[(fruit_df['Price'] > average_price) & (~fruit_df['Promotion'])]

print("Potential candidates for future promotions:")
print(potential_candidates[['Product', 'Price']])

#9
import pandas as pd
Employee_data = {'Employee': ['Alice', 'Bob', 'Charlie', 'David'],
                 'Department': ['HR', 'IT', 'Finance', 'IT'],
                 'Manager': ['John', 'Rachel', 'Emily', 'Rachel']}

Project_data = {'Employee': ['Alice', 'Charlie', 'Eve'],
                'Project': ['P1', 'P3', 'P2']}

employees_df = pd.DataFrame(Employee_data)
projects_df = pd.DataFrame(Project_data)

merged_df = pd.merge(projects_df, employees_df, on='Employee', how='left')

merged_df = merged_df.fillna('Unassigned')

department_overview = merged_df.groupby(['Department', 'Manager']).agg({'Project': lambda x: ', '.join(x)}).reset_index()

print(department_overview)

#10
import pandas as pd


data = {
    'Department': ['Electronics', 'Electronics', 'Clothing', 'Clothing', 'Home Goods'],
    'Salesperson': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Sales': [70000, 50000, 30000, 40000, 60000]
}
df = pd.DataFrame(data)


dep_sales = df.groupby('Department')['Sales'].sum()

dep_counts = df['Department'].value_counts()
average_sales_per_salesperson = dep_sales / dep_counts

ranked_departments = average_sales_per_salesperson.sort_values(ascending=False)

print("Average Sales per Salesperson in Each Department:")
print(average_sales_per_salesperson)

print("\nRanked Departments based on Average Sales per Salesperson:")
print(ranked_departments)