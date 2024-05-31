from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_table('../pythonProject/data/income.txt')
mean = df['income'].mean()

df['income'].fillna(mean, inplace=True)

a = df['income'].isnull().sum()


# 원하는 리스트를 여러개 짚어넣을 수 있음 리스트 리스트별 첫 항목을 뽑아서 리스트를 하나 더 만들어주는거임
x = np.column_stack([df['age'], df['age']**2, np.ones(len(df['age']))])
# x = np.column_stack([df['age'], df['age']**2]) # 위에 두개중에 r값 더 높은걸로 선택

model = sm.OLS(df['income'], x).fit()
print(model.summary())

# x1
# x2
# const 는 상수임 P