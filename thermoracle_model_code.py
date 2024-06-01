import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



df=pd.read_csv("dataroorkee.csv")
# print(df["Hour 0: Type_rain"])
y_label=df["Temperature (C)"]
# plt.plot(temp)
# plt.show()

mode_columns = {'Type_rain': ['Hour 0: Type_rain', 'Hour 1: Type_rain', 'Hour 2: Type_rain', 'Hour 3: Type_rain', 'Hour 4: Type_rain', 'Hour 5: Type_rain'],
                'Type_snow': ['Hour 0: Type_snow', 'Hour 1: Type_snow', 'Hour 2: Type_snow', 'Hour 3: Type_snow', 'Hour 4: Type_snow', 'Hour 5: Type_snow']}

for mode_attr, mode_cols in mode_columns.items():
    df[f'Hour 0: {mode_attr}'] = df[mode_cols].mode(axis=1)[0]

# Calculate mean for other attributes
other_attributes = ['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']

for attribute in other_attributes:
    cols = [f'Hour 0: {attribute}', f'Hour 1: {attribute}', f'Hour 2: {attribute}', f'Hour 3: {attribute}', f'Hour 4: {attribute}', f'Hour 5: {attribute}']
    df[f'Hour 0: {attribute}'] = df[cols].mean(axis=1)

# Select only the desired columns
selected_columns = ['Hour 0: Type_rain', 'Hour 0: Type_snow', 'Hour 0: Humidity', 'Hour 0: Wind Speed (km/h)', 'Hour 0: Wind Bearing (degrees)', 'Hour 0: Visibility (km)', 'Hour 0: Pressure (millibars)','Temperature (C)']

# Save the selected columns to a CSV file
df[selected_columns].to_csv('output.csv', index=False)
df_final=pd.read_csv("output.csv")
# print(scaled_data)

X = df.drop(columns=['Temperature (C)'])  # Drop the target variable column from features
y = df['Temperature (C)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Create polynomial features of degree 4
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Create a linear regression model
model = LinearRegression()

# Fit the model using the polynomial features
model.fit(X_train_poly, y_train)

# Make predictions on the test data
predictions = model.predict(X_test_poly)

# Evaluate the model (e.g., calculate mean squared error)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)


print(model.score(X_test_poly,y_test))
print(model.score(X_test,y_test))

accuracy=(y_test-predictions)/y_test
# print(sum(accuracy))
accuracy = model.score(X_test_poly, y_test)
print("Accuracy:", accuracy)


