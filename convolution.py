import tensorflow as tf
import numpy as np

(tranX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# 0에서 1으로 압축
tranX = tranX / 255.0
testX = testX / 255.0

# shape 변경 차원하나 늘려준거임
trainX = tranX.reshape((60000, 28, 28, 1))  # 칼라는 마지막이 3 이여야함
testX = testX.reshape((10000, 28, 28, 1))


model = tf.keras.models.Sequential([
    # 첫번째에 이미지 복사본 얼마나 생성할지 32면 32개의 복사본 생성하는거임 (3,3)은 커널의 크기임
    # padding은 이미지 변환안되게
    tf.keras.layers.Conv2D(32,  (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)), #(2,2)는 2 x 2 사이즈 커널인거임
    # tf.keras.layers.Dense(128, input_shape=(28, 28), activation='relu'),
    tf.keras.layers.Flatten(), # 2차원 이상의 데이터를 1차원으로 압축
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'), # 10개의 확율을 뱉어 주세요
])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(tranX, trainY, validation_data=(testX, testY), epochs=1)

model.evaluate(testX, testY)
