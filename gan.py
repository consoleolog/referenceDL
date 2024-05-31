from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
# 모든 파일들을 리스트로 담아줌
file_list = os.listdir('/content/img_align_celeba/img_align_celeba')

image_num_list = []

for i in file_list:
  # 이미지 숫자로 변환                  #convert('L')이면 흑백 컬러면 'RGB'     # 확대하는거임 주변에 여백 제거하는거라고 생각하면 될듯  (20,30)만큼짜르고 (160,180)만큼 사용
  숫자화된거 = Image.open('/content/img_align_celeba/img_align_celeba/'+i).convert('L').crop((20,30,160,180)).resize((64,64))
  image_num_list.append(np.array(숫자화된거)) # 나중에 전처리할 수도 있으니깐 넘파이로 변환한 다음에 집어넣는게 좋을듯?

plt.imshow(image_num_list[1])
plt.show()
print(image_num_list.shape) #(64,64)

# 노멀라이징
image_num_list = np.divide(image_num_list,255)
# Cov2D 는 4차원만 넣을 수 있어서 흑백은 3차원이여서 4차원으로 바꿔줘야함
image_num_list = image_num_list.reshape(10000,64,64,1)
print(image_num_list.shape) # (10000, 64, 64, 1)

#Discriminator
import tensorflow as tf

discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=[64,64,1]),
    tf.keras.layers.LeakyReLU(alpha=0.2), # leakyReLU 가 활성화 함수임 gan에서는 이게 더 좋다고 함 음의 값에 작은값을 곱해줌
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#Generator
generator = tf.keras.models.Sequential([
  tf.keras.layers.Dense(4 * 4 * 256, input_shape=(100,) ),
  tf.keras.layers.Reshape((4, 4, 256)),
  tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same'), #Transpose 이미지 크기 늘려주는거임 strides 가 크기 설정임
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same'), # (1,3) 뜻이 (64,64,1) 인거임
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')
])
generator.summary()

GAN = tf.keras.models.Sequential([ generator, discriminator ])

discriminator.compile(optimizer='adam', loss='binary_crossentropy')

discriminator.trainable = False

GAN.compile(optimizer='adam',loss='binary_crossentropy')



랜덤숫자 = np.random.uniform(-1,1,size=(8,100)) # 랜덤한 숫자 100개를 8세트 뽑아주세요


예측 = generator.predict(랜덤숫자)
print(예측.shape) # (8, 64, 64, 1)

for i in range(10):
  plt.subplot(2,5,i+1)
  plt.imshow(예측[i].reshape(64,64),cmap='gray') # 칼라면 64,64,3
  plt.axis('off')
plt.tight_layout()
plt.show()


# 진짜 사진들임
x_data = image_num_list
for j in range(300):
  print(f'지금 epoch 몇회냐면 {j}')
  for i in range(50000//128):
    if i % 100 == 0:
      print(f'지금 몇번째 batch냐면 {i}')
    # 진짜 사진 128장
    진짜사진들 = x_data[i*128:(i+1)*128]

    # 1로 마킹해야지 진짜 데이터니깐
    정답들 = np.ones(shape=(128,1))

    # 가짜 사진 50장
    랜덤숫자 = np.random.uniform(-1,1,size=(128,100))
    가짜사진들 = generator.predict(랜덤숫자)
    가짜사진들128장 = 가짜사진들[0:128]
    가짜정답 = np.zero(shape=(128,1))


    #discriminator training
    discriminator.train_on_batch( 진짜사진들 , 정답들 )
    discriminator.train_on_batch(가짜사진들128장, 가짜정답)

    #generator training
    GAN.train_on_batch(랜덤숫자, 정답들)