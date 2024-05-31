import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

키 = np.array([170, 180, 160, 165, 158, 176, 182, 172]).reshape((-1, 1))
몸무게 = [75, 81, 59, 70, 55, 78, 84, 72]
plt.scatter(키, 몸무게)
plt.show()

model = LinearRegression().fit(키, 몸무게)
print(model.score(키, 몸무게))  # r 값 데이터 관련도
print(model.intercept_)  # b 값 y 절편
print(model.coef_) # a 값 x 기울기

a = model.predict([[170]])
print(a)
plt.scatter(키, model.predict(키))
plt.show()
