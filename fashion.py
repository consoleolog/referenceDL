import tensorflow as tf

(tranX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_shape=(28, 28), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Flatten(), # 2차원 이상의 데이터를 1차원으로 압축
    tf.keras.layers.Dense(10, activation='softmax'), # 10개의 확율을 뱉어 주세요
])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(tranX, trainY, validation_data=(testX, testY), epochs=1)
