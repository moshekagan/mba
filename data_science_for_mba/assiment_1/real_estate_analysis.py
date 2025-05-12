#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Importing data
df = pd.read_csv("HousePricesHW1.csv")
df.head()

# Generating Histogram & Density Plot
plt.figure(figsize=(10, 6))
sns.histplot(df["Price"], kde=True, bins=30)

plt.title("Distribution of Apartment Prices")
plt.xlabel("Price (NIS)")
plt.ylabel("Frequency")

plt.grid(False)
plt.show()

# Summary aggregate statistics
mean_price = df["Price"].mean()
median_price = df["Price"].median()
std_price = df["Price"].std()

print("------ Summary Statistics for Apartment Prices ------")
print(f"Mean: {mean_price:,.0f} NIS")
print(f"Median: {median_price:,.0f} NIS")
print(f"Standard Deviation: {std_price:,.0f} NIS")


# Generating Box Plot to compare the number of stores in the area with and without dog parks
plt.figure(figsize=(8, 6))
sns.boxplot(x="DogParkInd", y="NumStores", data=df)

plt.title("Number of Food Stores by Dog Park Presence")
plt.xlabel("Dog Park Present (1 = Yes, 0 = No)")
plt.ylabel("Number of Food Stores (within 100m)")
plt.grid(False)
plt.show()

# Building linear regression model with all variables
# dependent variable
y = df["Price"]

# independent variables for model m1
X = df[["MtrsToBeach", "SqMtrs", "Age", "NumStores", "DogParkInd", "SchoolScores"]]
X = sm.add_constant(X)

# building the model
model_m1 = sm.OLS(y, X).fit()
print("\n\n------ Model 1 Summary ------\n")
print(model_m1.summary())

# Calculating the price difference for 18 square meters
add_18_meter = model_m1.params["SqMtrs"] * 18
print(f"\nWhen adding 18 square meters, the price difference is: {add_18_meter} NIS")

# Building linear regression model with metrics to beach, square meters, and age
# dependent variable
y = df["Price"]
# independent variables for model m2
X_m2 = df[["MtrsToBeach", "SqMtrs", "Age"]]
X_m2 = sm.add_constant(X_m2)

# building the model
model_m2 = sm.OLS(y, X_m2).fit()
print("\n\n------ Model 2 Summary ------\n")
print(model_m2.summary())

# Calculating the price difference for 18 square meters
add_18_meter = model_m2.params["SqMtrs"] * 18
print(f"\nWhen adding 18 square meters, the price difference is: {add_18_meter} NIS")

# Building linear regression model without the number of stores
y = df["Price"]
X_no_stores = df[["MtrsToBeach", "SqMtrs", "Age", "DogParkInd", "SchoolScores"]]
X_no_stores = sm.add_constant(X_no_stores)

model_no_stores = sm.OLS(y, X_no_stores).fit()

print("\n\n------ Model without Number of Stores Summary ------\n")
print(model_no_stores.summary())
