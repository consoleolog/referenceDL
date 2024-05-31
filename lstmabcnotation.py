import tensorflow as tf
import numpy as np
import pandas as pd


text = open('./data/pianoabc.txt','r').read()

#Bag of word
uniqueText = list(set(text))
uniqueText.sort()



# 문자를 넣으면 숫자로 되는거임
text_to_num = {}

# 숫자를 넣으면 문자를 뱉어줌
num_to_text = {}

for i, data in enumerate(uniqueText):
    text_to_num[data] = i
    num_to_text[i] = data

num_list = []
# text를 돌면서 문자들을 다 숫자로 바꿔야함
for string in text:
    num_list.append(text_to_num[string])

trainX = []
trainY = []

for i in range(0, len(num_list)-25):
    trainX.append(num_list[i:i+25])
    trainY.append(num_list[i+25])

trainX = tf.one_hot(trainX, len(uniqueText))
trainY = tf.one_hot(trainY, len(uniqueText))

valX = []
valY = []

#print(trainX[0:2]) # 얘로 인풋 shape 확인

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, input_shape=(25, 31)),
    tf.keras.layers.Dense(len(uniqueText), activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX,trainY,batch_size=64,epochs=1, verbose=2)
model.save('./saved_models/lstmabcnotation')






