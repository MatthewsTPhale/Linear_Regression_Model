import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def load_data(file_path):
    base_dir = os.path.dirname(__file__)  # folder where main.py is located
    full_path = os.path.join(base_dir, "data", file_path)
    bike = pd.read_csv(full_path, index_col="dteday", parse_dates=True)
    return bike

# Load the data
bikes = load_data("hour.csv")

# Rename the 'cnt' column to 'total_rentals'
bikes.rename(columns={"cnt": "total_rentals"}, inplace=True)

# Display the first 10 rows of the DataFrame
print(bikes.head(10))

# Scatter plot
# bikes.plot(
#     kind="scatter",
#     x="temp",
#     y="total_rentals",
#     alpha=0.5,                 # transparency so points don’t overlap too heavily
#     figsize=(8,6),             # larger figure
#     title="Bike Rentals vs. Temperature"
# )

# # Add axis labels
# plt.xlabel("Normalized Temperature (0–1 scale)")
# plt.ylabel("Total Rentals")
# plt.show()

# Simple linear regression
features = ["temp"]
X = bikes[features]
print(X.head(10))
Y = bikes["total_rentals"]
print(Y.head(10))   

# Fit the model
model = LinearRegression()
model.fit(X, Y)

print("Intercept: ",model.intercept_)
print("Coefficient: ",model.coef_) 

# Prediction of specific temperatures
X_new = pd.DataFrame({"temp": [0.2, 1.0, 0.3, 0.9]})
predictions = model.predict(X_new)
print(predictions)  # Predict rentals at 20% and 100% of normalized temperature

# Manual prediction for temp = 0.2
test = model.coef_[0]*X_new["temp"][0] + model.intercept_
print(test)

# ------Plotting --------
plt.figure(figsize=(10,6))

# Scatter plot
plt.scatter(bikes["temp"], bikes["total_rentals"], alpha=0.5, label="ActuaL Rentals")

# Regression line
plt.plot(bikes["temp"], model.predict(X), color="red", linewidth=2, label="Regression Line")

# Highlight predictions
plt.scatter(X_new['temp'], predictions, color='green', s=100, label="Predictions")

# Lables ad Title
plt.xlabel("Normalized Temperature (0–1 scale)")
plt.ylabel("Total Rentals")
plt.title("Bike Rentals vs. Temperature with Regression Line")
plt.legend()
plt.show()