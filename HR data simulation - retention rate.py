# create random data for retention rate by department over a 5 year period

import random
from typing import Dict, List, Any

import pandas as pd
# create a list of departments
departments = ['HR', 'Finance', 'R&D', 'Marketing', 'Sales', 'IT', 'Legal', 'Operations', 'Customer Service', 'Admin']
#  create a list of years
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

#create a dictionary to store the data
data: dict[str, list[float] | list[Any]] = {}
# 4. loop through the departments
for department in departments:
#  create a list to store the retention rate for the department
    retention_rate = []
#loop through the years
    for month in months:
# generate a random retention rate between 0 and 1 for each month and department
        rate = random.uniform(0, 1)
# append the retention rate to the list
        retention_rate.append(rate)
# add the list of retention rates to the dictionary
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

#create a model to determine if the rention  rate is anomalous
import numpy as np

# loop through the departments
for department in departments:
# create a list to store the anomaly score
    anomaly_score = []
# calculate the mean of the retention rate for the department
    mean = np.mean(data[department])
# calculate the standard deviation of the retention rate for the department
    std = np.std(data[department])
# calculate the anomaly score for the department
    score = (data[department] - mean) / std
# loop through the retention rate for the department
    for rate in data[department]:
    # calculate the anomaly score for the retention rate
        score = (rate - mean) / std
    # append the anomaly score to the list
        anomaly_score.append(score)
# add the anomaly score to the dictionary
    data[department + '_anomaly'] = anomaly_score
# print the data in pandas dataframe to inspect the data
data_df = pd.DataFrame(data, index=months)
#print(data_df)


# Determine if the retention rate for a specific department is anomalous for a specific month
# set the department to review
department = 'HR'
# loop through the months
for month in months:
# get the retention rate for the department and month
    rate = data[department][month - 1]
# get the anomaly score for the department and month
    score = data[department + '_anomaly'][month - 1]
# set the threshold for the anomaly score for the department less than -1
    threshold = -0.5
# determine if the anomaly score is above the threshold
    if score < threshold:
    # print a message if the anomaly score is above the threshold
        print(f'{department} has an anomalous retention rate for month {month}')
    else:
    # print a message if the anomaly score is not above the threshold
        print(f'{department} does not have an anomalous retention rate for month {month}')
# print the retention rate and anomaly score for the department for each month
for month in months:
    print(f'{department} retention rate for month {month}: {data[department][month - 1]}')
    print(f'{department} anomaly score for month {month}: {data[department + "_anomaly"][month - 1]}')
    print(f'{department} mean retention rate: {np.mean(data[department])}')
    print(f'{department} standard deviation of retention rate: {np.std(data[department])}')
    print('\n')

'''
#note you must comment out the previouly block of code to run the regression model

#create a regression model to predict the retention rate for a specific department for month 13
from sklearn.linear_model import LinearRegression
# set the department to predict
department = 'HR'
# get the retention rate for the department
y = data[department]
# create a list of months
X = np.array(months).reshape(-1, 1)
# create a linear regression model
model = LinearRegression()
# fit the model to the data
model.fit(X, y)
# predict the retention rate for month 13
prediction = model.predict([[13]])
# print the predicted retention rate for month 13
print(f'The predicted retention rate for {department} in month 13 is: {prediction[0]}')
'''




