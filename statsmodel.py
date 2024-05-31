import statsmodels.api as sm
import numpy as np


키 = np.array([170, 180, 160, 165, 158, 176, 182, 172]).reshape((-1, 1))
몸무게 = [75, 81, 59, 70, 55, 78, 84, 72]

# model = sm.OLS(y, x).fit()
model = sm.OLS(몸무게,키).fit()
print(model.summary())
# R-squared 값 r 제곱값
# Prob x 의 계수가 0일 확률 08 이면 0이 8개인거임 거의 0이라는거지

# x1 coef x 계수 std err 표준 요차
# P>|t| 가 작을 수록 좋음 대충 0.005 이하면 좋은거임
# [0.0023 0.975] 신뢰구간

