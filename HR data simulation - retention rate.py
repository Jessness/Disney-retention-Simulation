# create random data for retention rate by department over a 5 year period

import random
from typing import Dict, List, Any

import pandas as pd
departments = ['HR', 'Finance', 'R&D', 'Marketing', 'Sales', 'IT', 'Legal', 'Operations', 'Customer Service', 'Admin']
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
data: dict[str, list[float] | list[Any]] = {}
for department in departments:
    retention_rate = []
    for month in months:
        rate = random.uniform(0, 1)
        retention_rate.append(rate)
        data[department] = retention_rate

# print the data in pandas dataframe to inspect the data
#data = pd.DataFrame(data, index=months)
#print(data)

# create a visualization of a time series plot of the retention rate by department
'''
import matplotlib.pyplot as plt
# set the size of the plot
plt.figure(figsize=(10, 6))
# loop through the departments
for department in departments:
# plot the retention rate for the department
    plt.plot(months, data[department], label=department)
# add a title to the plot
plt.title('Retention Rate by Department')
# add labels to the x and y axes
plt.xlabel('Month')
plt.ylabel('Retention Rate')
# add a legend to the plot
plt.legend()
# display the plot
plt.show()
'''

#create an anomaly score to determine if the rention  rate is anomalous
import numpy as np
for department in departments:
    anomaly_score = []
    mean = np.mean(data[department])
    std = np.std(data[department])
    score = (data[department] - mean) / std
    for rate in data[department]:
        score = (rate - mean) / std
        anomaly_score.append(score)
    data[department + '_anomaly'] = anomaly_score
data_df = pd.DataFrame(data, index=months)
#print(data_df)


# Determine if the retention rate for a specific department is anomalous for a specific month
# set the department to review
department = 'HR'
for month in months:
    rate = data[department][month - 1]
    score = data[department + '_anomaly'][month - 1]
    threshold = -0.5
    if score < threshold:
        print(f'{department} has an anomalous retention rate for month {month}')
    else:
        print(f'{department} does not have an anomalous retention rate for month {month}')
for month in months:
    print(f'{department} retention rate for month {month}: {data[department][month - 1]}')
    print(f'{department} anomaly score for month {month}: {data[department + "_anomaly"][month - 1]}')
    print(f'{department} mean retention rate: {np.mean(data[department])}')
    print(f'{department} standard deviation of retention rate: {np.std(data[department])}')
    print('\n')


#create a regression model to predict the retention rate for a specific department for month 13
from sklearn.linear_model import LinearRegression
department = 'HR'
y = data[department]
X = np.array(months).reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[13]])
print(f'The predicted retention rate for {department} in month 13 is per regression model: {prediction[0]}')


#build a machine learning model to predict the retention rate for a specific department for month 13
from sklearn.ensemble import RandomForestRegressor
department = 'HR'
y = data[department]
X = np.array(months).reshape(-1, 1)
model = RandomForestRegressor()
model.fit(X, y)
prediction = model.predict([[13]])
print(f'The predicted retention rate for {department} in month 13 is per random forest model: {prediction[0]}')

#validate the regression and random forest models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
department = 'HR'
y = data[department]
X = np.array(months).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'The mean squared error of the linear regression model is: {mse}')
model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'The mean squared error of the random forest model is: {mse}')



