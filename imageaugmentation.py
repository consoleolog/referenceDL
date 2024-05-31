import tensorflow as tf
import numpy as np
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
if not os.path.exists('dataset'):
    os.mkdir("dataset")
    os.mkdir("../recommendProject/dataset/image_1/cat")
    os.mkdir("dataset/dog")

for i in os.listdir('./catdog/train/'):
    if 'cat' in i:
        shutil.copyfile('./catdog/train/'+ i, './dataset/cat/'+ i)
    if 'dog' in i:
        shutil.copyfile('./catdog/train/'+ i, './dataset/dog/'+ i)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset',
    image_size=(64, 64),
    # 그냥 학습을하지않고 한번에 집어넣는게 아니라 32개씩 집어넣고 w 값 계산하고 그런식으로하는거임
    batch_size=32,
    subset="training", # 데이터 이름 이건 트레이닝 데이터
    validation_split=0.2, # 데이터를 0.2 개로 쪼개는거임,
    seed=123,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset',
    image_size=(64, 64),
    # 그냥 학습을하지않고 한번에 집어넣는게 아니라 32개씩 집어넣고 w 값 계산하고 그런식으로하는거임
    batch_size=32,
    subset="training",  # 데이터 이름 이건 트레이닝 데이터
    validation_split=0.2,  # 데이터를 0.2 개로 쪼개는거임,
    seed=123,
)
# def normalize(i, answer):
#     i = tf.cast(i/255.0, tf.float32)
#     return i, answer
#
# train_ds = train_ds.map(normalize)
#
# val_ds = val_ds.map(normalize)

img_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20, # 회전
    zoom_range=0.15, # 확대
    width_shift_range=0.2, # 이동
    height_shift_range=0.2,
    shear_range=0.15,  # 굴정
    horizontal_flip=True, # 가로 반전
    fill_mode="nearest"
)
# 얘를 model.fit 에 넣으면 됨
train_generator = img_generator.flow_from_directory(
    'dataset/cat',
    class_mode='binary', # 두개면 binary, 몇 개 더면 categorical
    shuffle=True,
    seed=123,
    color_mode='rgb',
    batch_size=64,
    target_size=(64, 64)
)

검증용생성기 = ImageDataGenerator(rescale=1./255)

val_generator = img_generator.flow_from_directory(
    'dataset/cat',
    class_mode='binary',
    shuffle=True,
    seed=123,
    color_mode='rgb',
    batch_size=64,
)


model = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(64,64,3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
    # 첫번째에 이미지 복사본 얼마나 생성할지 32면 32개의 복사본 생성하는거임 (3,3)은 커널의 크기임
    # padding은 이미지 변환안되게
    tf.keras.layers.Conv2D(32,  (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)), #(2,2)는 2 x 2 사이즈 커널인거임
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Dense(128, input_shape=(28, 28), activation='relu'),
    tf.keras.layers.Flatten(), # 2차원 이상의 데이터를 1차원으로 압축
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=1)