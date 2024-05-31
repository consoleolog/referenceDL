import tensorflow as tf
import numpy as np
import os
import shutil
from tensorflow.keras.applications import InceptionV3

#from tensorflow.python.keras.applications import InceptionV3

inception_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)
inception_model.load_weights('./models/inception_v3_weights.h5')

inception_model.summary()

for inception_layer in inception_model.layers:
    inception_layer.trainable = False

# 파인 튜닝 하는법
unfreeze = False
for i in inception_model.layers:
    if i.name == 'mixed6': # mixed 6부터 다시 트레이닝
        unfreeze = True
    if unfreeze == True :
        i.trainable = True

last_layer = inception_model.get_layer('mixed7')

layer_1 = tf.keras.layers.Flatten()(last_layer.output)
dense_1 = tf.keras.layers.Dense(1024, activation='relu')(layer_1)
drop_1 = tf.keras.layers.Dropout(0.2)(dense_1)
output = tf.keras.layers.Dense(1, activation='sigmoid')(drop_1)

model = tf.keras.Model(inputs=inception_model.input, output=output)

# 파인튜닝했을때 러닝 레이트 아주 조금씩만 설정
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



exit()
if not os.path.exists('dataset'):
    os.mkdir("dataset")
    os.mkdir("dataset/cat")
    os.mkdir("dataset/dog")

for i in os.listdir('./catdog/train/'):
    if 'cat' in i:
        shutil.copyfile('./catdog/train/' + i, './dataset/cat/' + i)
    if 'dog' in i:
        shutil.copyfile('./catdog/train/' + i, './dataset/dog/' + i)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './dataset/',
    image_size=(150, 150),
    batch_size=64,
    subset='training',
    validation_split=0.2,
    seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './dataset/',
    image_size=(150, 150),
    batch_size=64,
    subset='validation',
    validation_split=0.2,
    seed=1234
)

print(train_ds)


def 전처리함수(i, 정답):
    i = tf.cast(i / 255.0, tf.float32)
    return i, 정답


train_ds = train_ds.map(전처리함수)
val_ds = val_ds.map(전처리함수)

model = tf.keras.models.Sequential([
    # 첫번째에 이미지 복사본 얼마나 생성할지 32면 32개의 복사본 생성하는거임 (3,3)은 커널의 크기임
    # padding은 이미지 변환안되게
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),  #(2,2)는 2 x 2 사이즈 커널인거임
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Dense(128, input_shape=(28, 28), activation='relu'),
    tf.keras.layers.Flatten(),  # 2차원 이상의 데이터를 1차원으로 압축
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),  # 10개의 확율을 뱉어 주세요
])
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=1)
