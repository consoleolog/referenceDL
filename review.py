import urllib.request
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# urllib.request.urlretrieve('https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt', './data/shopping.txt')

raw = pd.read_table('./data/shopping.txt', names=['rating', 'review'])

# 라벨링 작업
raw['label'] = np.where(raw['rating'] > 3, 1, 0)

# 특수문자 제거
raw['review'] = raw['review'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]', '')

print(raw.isnull().sum())

# 리뷰 데이터 중복 제거
raw.drop_duplicates(subset=['review'], inplace=True)

# Bag of words
uniqueText = raw['review'].tolist()
uniqueText = ''.join(uniqueText)
uniqueText = list(set(uniqueText))
uniqueText.sort()

# 토크나이저 설정 char_level=True 면 글자 단위를 정수로 변환 False 면 단어 단위(띄어쓰기) oov_token은 테스트 할 때 처음보는 단어가 나오면 어떤걸로 바꿀지 out of vocabluary 줄임말인듯
tokenizer = Tokenizer(char_level=True, oov_token='<OOV>')

text_list = raw['review'].tolist()

# fit_one_tesxts(긴 문자리스트) 쓰면 word_index 를 생성해주는데 이게 문자를 숫자로 바꾼 딕셔너리인거임
# print(tokenizer.word_index)
tokenizer.fit_on_texts(text_list)

# texts_to_sequences 쓰면 문자 리스트를 정수로 변환해줌
train_seq = tokenizer.texts_to_sequences(text_list)

# 정답 데이터
Y = raw['label'].tolist()

# 제일 긴 문장을 찾아
# print(raw.describe())
# 길이 열을 하나 더 추가 하자
raw['length'] = raw['review'].str.len()

# 최대 길이 설정
X = pad_sequences(train_seq, maxlen=100)

trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 16), #원핫인코딩하는대신
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=1, validation_data=(valX, valY))


