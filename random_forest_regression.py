# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Final Train.csv')
dataset_test = pd.read_csv('Final Test.csv')
y_train = dataset.iloc[:, 36].values
dataset.drop('SalePrice', axis = 1, inplace = True)
X_train = dataset.iloc[:, :].values
X_test = dataset_test.iloc[:, :].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(np.array(y_train).reshape(1455,1))

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)

"""
#Calculating Error
s = 0
for i in range(1459):
        s += abs(submission.iloc[i, 1] - y_pred[i])
print('Error is: ',s)"""

# Creating Submission File
arr = np.array(list(range(1461, 2920)))
final_id = pd.DataFrame(arr, columns = ['Id'])
final_sale_price = pd.DataFrame(y_pred, columns = ['SalePrice'])
final = pd.concat([final_id, final_sale_price], axis = 1)
final.to_csv('result_random_forest.csv', index = False)