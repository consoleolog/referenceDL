import pandas as pd
import numpy as np

df = pd.read_csv("./data/credit.csv")

# 1과 가까울수록 비례 01 과 가까울수록 반비례

yes_m = df.query("성별=='M' and 기혼=='Married'")
no_m = df.query("성별=='M' and 기혼=='Sigle'")
print(yes_m['사용금액'].mean() > no_m['사용금액'].mean())

# 소득이 높을 수록 사용금액이 높은가
# 소득과 사용금액이 비례하냐

print(df.groupby('소득').mean())