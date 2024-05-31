import tensorflow as tf
import numpy as np

text = open('./data/pianoabc.txt','r').read()

model = tf.keras.models.load_model('./saved_models/model')

uniqueText = list(set(text))
uniqueText.sort()

text_to_num = {}
num_to_text = {}

for i, data in enumerate(uniqueText):
    text_to_num[data] = i
    num_to_text[i] = data

num_list = []
for i in text:
    num_list.append(text_to_num[i])

music = []

# 첫 입력값 만들기
first_input = num_list[117:117+25]
first_input = tf.one_hot(first_input, len(num_to_text))
first_input = tf.expand_dims(first_input, axis=0)

for i in range(200):
    # predict로 다음문자 예측
    predict_value = model.predict(first_input)
    # predict_value = np.argmax(predict_value[0]) # 31개의 확률 중에서 제일 큰 확률을 골라주는거임
    # print(predict_value)

    # 예측값 중에 랜덤한 걸 뽑아줌 근데 문자임 원래 쟤는 확률을 골라주는애인데
    predict_value = np.random.choice(uniqueText, 1, p=predict_value[0])
    predict_value = ''.join(predict_value)
    predict_value = text_to_num[predict_value]

    # 예측한 다음문자 [] 에 저장하기
    music.append(predict_value)

    # 첫입력값 앞에 짜르기
    next_input = first_input.numpy()[0][1:] # first input은 이미 원핫인코딩이 되어있는거임

    one_hot_num = tf.one_hot(predict_value, len(num_to_text))

    first_input = np.vstack([next_input, one_hot_num.numpy()])
    first_input = tf.expand_dims(first_input, axis=0)

music_text = []

for i in music:
    music_text.append(num_to_text[i])

print(''.join(music_text))

# 첫 입력값 만들기
# predict로 다음문자 예측
# 예측한 다음문자 [] 에 저장하기
# 첫입력값 앞에 짜르기
# 예측한 다음 문자를 뒤에 넣기
# 원핫인코딩하기

