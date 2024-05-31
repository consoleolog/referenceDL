import tensorflow as tf
import numpy as np
import os
import shutil

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
def normalize(i, answer):
    i = tf.cast(i/255.0, tf.float32)
    return i, answer

train_ds = train_ds.map(normalize)
val_ds = val_ds.map(normalize)
model = tf.keras.models.Sequential([
    # 첫번째에 이미지 복사본 얼마나 생성할지 32면 32개의 복사본 생성하는거임 (3,3)은 커널의 크기임
    # padding은 이미지 변환안되게
    tf.keras.layers.Conv2D(32,  (3, 3), padding='same', activation='relu', input_shape=(64,64,3)),
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
    tf.keras.layers.Dense(1, activation='sigmoid'),# 10개의 확율을 뱉어 주세요
])
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=1)