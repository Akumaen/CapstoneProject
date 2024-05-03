# Supressing the warning messages
import warnings

warnings.filterwarnings("ignore")
from typing import Any
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Reading the CSV file into a pandas DataFrame
car_data = pd.read_csv('car_price_prediction.csv')

# Displaying the shape of the dataset before removing duplicates
print('Shape before deleting duplicate values:', car_data.shape)

# Removing duplicate rows if any
car_data = car_data.drop_duplicates()

# Displaying the shape of the dataset after removing duplicates
print("Shape After deleting duplicate values:", car_data.shape)

# Visualising the distribution of Target variable Price
plt.figure(figsize=(10, 6))
sns.histplot(car_data['Price'])
plt.title("Distribution of Price")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Basic Exploratory Data Analysis
print(" 1. View sample rows of the data (head and tail)")
print("Sample Rows (Head):")
print(car_data.head(10))
print("Sample Rows (Tail):")
print(car_data.tail(10))

print(" 2. Summarized information of the data (info)")
print("Summarized Information:")
print(car_data.info())

print(" 3. Descriptive statistics of the data (describe)")
print("Descriptive Statistics:")
print(car_data.describe(include='all'))

print(" 4. Unique values for each column (nunique)")
print("Unique Values per Column:")
print(car_data.nunique())

# Rejecting unwanted columns
car_data = car_data.drop('ID', axis=1)
car_data = car_data.drop('Model', axis=1)
car_data = car_data.drop('Leather interior', axis=1)
car_data = car_data.drop('Category', axis=1)
car_data = car_data.drop('Cylinders', axis=1)
car_data = car_data.drop('Gear box type', axis=1)
car_data = car_data.drop('Drive wheels', axis=1)
car_data = car_data.drop('Doors', axis=1)
car_data = car_data.drop('Wheel', axis=1)
car_data = car_data.drop('Color', axis=1)

# Visual exploratory data analysis using histograms
plt.figure(figsize=(12, 12))
plt.subplot(331)
sns.histplot(car_data['Levy'])
plt.subplot(332)
sns.histplot(car_data['Prod. year'])
plt.subplot(333)
sns.histplot(car_data['Engine volume'])
plt.subplot(334)
sns.histplot(car_data['Mileage'])
plt.subplot(335)
sns.histplot(car_data['Airbags'])
plt.show()

# Visual exploratory data analysis using bar charts
plt.figure(figsize=(12, 6))
plt.subplot(121)
sns.countplot(x='Manufacturer', data=car_data)
plt.title("Manufacturer")
plt.subplot(122)
sns.countplot(x='Fuel type', data=car_data)
plt.title("Fuel Type")
plt.show()

# Feature selection based on data distribution
selected_features = ['Price','Levy', 'Manufacturer', 'Prod. year', 'Fuel type', 'Engine volume', 'Mileage', 'Airbags']
selected_data: Series | None | DataFrame | Any = car_data[selected_features]

# Finding the number of missing values for each selected column
missing_values = car_data[selected_features].isnull().sum()
print("Missing values per column:")
print(missing_values)

# Removal of outliers and missing values
selected_data = car_data[selected_features].dropna()

# Visualize the distribution of the target variable after removing outliers and missing values
plt.figure(figsize=(10, 6))
sns.histplot(selected_data['Price'])
plt.title("Distribution of Price after outliers have been removed")
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Data conversion to numeric values
numerical_data: DataFrame = pd.get_dummies(selected_data)

# Split the dataset into features (X) and target (y) variables
X = numerical_data.drop('Price', axis=1)
y = numerical_data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model using the training set
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model using the testing set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

