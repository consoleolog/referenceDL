import pandas as pd
from pandas_datareader import data
import yfinance as yfin
import matplotlib.pyplot as plt

yfin.pdr_override()
df = data.get_data_yahoo('AAPL', start='2024-01-01', end='2024-05-01')

df2 = data.get_data_yahoo('005930.KS', start='2024-01-01', end='2024-05-01')
print(df2)
print(df2['Close'])

#컬럼 타입
# df['컬럼명'] = df['컬럼명'].astype(float) #타입 변환한걸 덮어쓰기

df2['rolling5'] = df2['Close'].rolling(5).mean() # 5개씩 평균을 내줌
df2['rolling20'] = df2['Close'].rolling(20).mean()
df2['Close'].rolling(5).sum() # 5개씩 합을 내줌
aws = data.get_data_yahoo('AMZN', start='2023-01-01', end='2023-12-31')

sam = [50000, 60000, 75000, 70000]
lg = [30000, 40000, 50000, 35000]
year = [2018, 2019, 2020, 2021]
plt.plot( year , sam, color='blue')
plt.plot( year , lg, color='red')
plt.xlabel('년도')
plt.ylabel('수익')
plt.legend(['삼성','LG'])
plt.show()






# aws['rolling20'] = aws['Close'].rolling(20).mean()
# aws['rolling60'] = aws['Close'].rolling(60).mean()
#
# plt.plot(aws.index, aws['rolling20'])
# plt.plot(aws.index, aws['rolling60'])
# plt.plot(aws.index, aws['Close'])
#
# plt.show()

# plt.show(df2.index, df2['Close'])
# plt.show()

# plt.show(x축 data, y축 data)
# plt.xlabel('작명')
# plt.ylabel('작명')



exit()
plt.figure(figsize=(10, 10))
plt.plot(df.index, df['Close'], color='crimson')
plt.plot(df.index, df['Open'], color='blue')
plt.xlabel('time')
plt.ylabel('price')
plt.legend(['apple'])
plt.show()

plt.bar([1,2,3],[2,3,4,])
plt.show()

plt.pie([2,3,4,5], labels=['apple','orange','pear','banana'])
plt.show()

plt.hist([160,165,166,167,168,169])
plt.show()

math = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
eng = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
plt.scatter(math, eng)
plt.show()

# stackplot()
plt.stackplot([1,2,3],[10,20,30],[30,20,50])
plt.show()