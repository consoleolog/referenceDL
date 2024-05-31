from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_table('../pythonProject/data/income.txt')
mean = df['income'].mean()

df['income'].fillna(mean, inplace=True)

a = df['income'].isnull().sum()

# y = ax + bx^2 + c

plt.scatter(df['age'], df['income'])
plt.show()

# return 에 함수 적으면됨
def 함수(x, a, b, c):
    return a*x + b*x**2 + c

#opt 는 a,b,c 값 cov 는 공분산값임
opt, cov = curve_fit(함수, df['age'], df['income'])
a, b, c = opt
print(opt) #[73.28756549 -8.43524542 -7.01431757]

#y = 73x -8x^2 - 7
x = np.array([1, 2, 3, 4, 5, 6])

plt.scatter(df['age'], df['income'])
plt.plot(x, 함수(x,a,b,c))
plt.show()
