import pandas as pd
import statsmodels.api as sm
import numpy as np

df = pd.read_csv('../pythonProject/data/california_housing.csv')

year = df['year']
rooms = df['rooms']
bedrooms = df['bedrooms']
price = df['price']

model = sm.OLS(price, df[['year', 'rooms', 'bedrooms']]).fit()
print(model.summary())

a = model.predict([
    [20,1000,200],
    [30,1000,200]
])
print(a)