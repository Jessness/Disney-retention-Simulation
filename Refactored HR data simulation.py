import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Department:
    def __init__(self, name: str):
        self.name = name
        self.retention_rate = [random.uniform(0, 1) for _ in range(12)]
        self.anomaly_score = self.calculate_anomaly_score()

    def calculate_anomaly_score(self):
        mean = np.mean(self.retention_rate)
        std = np.std(self.retention_rate)
        return [(rate - mean) / std for rate in self.retention_rate]

    def is_anomalous(self, month: int, threshold: float = -0.5):
        return self.anomaly_score[month - 1] < threshold

class Model:
    def __init__(self, model_type: str, department: Department):
        self.model_type = model_type
        self.department = department
        self.model = LinearRegression() if model_type == 'regression' else RandomForestRegressor()
        self.X = np.array(list(range(1, 13))).reshape(-1, 1)
        self.y = department.retention_rate
        self.model.fit(self.X, self.y)

    def predict(self, month: int):
        return self.model.predict([[month]])

    def validate(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

departments = [Department(name) for name in ['HR', 'Finance', 'R&D', 'Marketing', 'Sales', 'IT', 'Legal', 'Operations', 'Customer Service', 'Admin']]
models = [Model(model_type, department) for model_type in ['regression', 'random_forest'] for department in departments]


for department in departments:
    for month in range(1, 13):
        if department.is_anomalous(month):
            print(f'{department.name} has an significant decline in retention rate for month {month}')
        else:
            print(f'{department.name} does not have an significant decline in retention rate for month {month}')
        print(f'{department.name} retention rate for month {month}: {department.retention_rate[month - 1]}')
        print(f'{department.name} anomaly score for month {month}: {department.anomaly_score[month - 1]}')
        print(f'{department.name} mean retention rate: {np.mean(department.retention_rate)}')
        print(f'{department.name} standard deviation of retention rate: {np.std(department.retention_rate)}')
        print('\n')


for model in models:
    print(f'The predicted retention rate for {model.department.name} in month 13 is per {model.model_type} model: {model.predict(13)[0]}')
    print(f'The mean squared error of the {model.model_type} model is: {model.validate()}')

