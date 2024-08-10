import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
import numpy as np

from tensorflow.keras.layers import (LeakyReLU, BatchNormalization, Reshape, Flatten, Input, Dense)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

image_folder = "../SGAN/photo_move/"

img_rows = 64  # 28
img_cols = 64  # 28
channels = 1  #

img_shape = (img_rows, img_cols, channels)
output_dir = "../sksk/generated_images/"

image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]


# 이미지를 불러와서 전처리하는 함수
def load_and_preprocess_image(image):
    image = image.convert('L')  # 흑백 이미지로 변환
    image = image.resize((img_cols, img_rows))
    image = np.array(image) / 127.5 - 1.0  #
    image = np.expand_dims(image, axis=-1)
    return image


# 이미지들을 불러와 전처리하여 저장하는 리스트

# 이미지와 레이블을 저장할 리스트 초기화
images = []
labels = []

# 현재 경로 내의 파일들을 순회하며 이미지 파일을 찾아 리스트에 추가
for filename in os.listdir(image_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):  # 지원하는 이미지 확장자 추가
        # 이미지 파일 열기
        image = Image.open(os.path.join(image_folder, filename))
        image = load_and_preprocess_image(image)

        # RGB 형태로 변환하여 리스트에 추가
        images.append(image)

        # 파일명에서 레이블 정보 추출 (파일명 예시: 'train_00229_002.png')
        label = filename[-5]

        label= int(label)
        labels.append(label)



print(len(images))
print(labels)
# labels을 numpy 배열로 변환

# 생성자 모델
generator = Sequential([
    Dense(128, input_dim=100),
    LeakyReLU(alpha=0.01),
    BatchNormalization(),
    Dense(64 * 64 * 1, activation='tanh'),  # 출력 크기를 (64, 64, 1)에 맞게 조정
    # Dense(784, activation='tanh'),
    Reshape((64, 64, 1))  # 이미지는 흑백이므로 채널이 1
])

# 분류자 모델
discriminator = Sequential([
    Flatten(input_shape=(64, 64, 1)),
    Dense(128),
    LeakyReLU(alpha=0.01),
    Dense(1, activation='sigmoid')
])

# 생성자와 분류자를 결합하여 SGAN 모델 생성
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
discriminator.trainable = False  # 판별자는 훈련 중 업데이트되지 않도록 고정

z = Input(shape=(100,))
img = generator(z)
validity = discriminator(img)

sgan = Model(z, validity)
sgan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))


# 반지도 학습을 위한 레이블 있는 데이터와 없는 데이터 나누기
# labeled_indices = np.where(y_train < 6)[0]
# unlabeled_indices = np.where(y_train >= 6)[0]
# num_labeled = len(labeled_indices)
# num_unlabeled = len(unlabeled_indices)

# 반지도 학습을 위한 레이블 있는 데이터와 없는 데이터 나누기
labeled_indices = np.where(np.array(labels, dtype=int) < 6)[0]
unlabeled_indices = np.where(np.array(labels, dtype=int) >= 6)[0]

num_labeled = len(labeled_indices)
num_unlabeled = len(unlabeled_indices)
print(num_labeled)
print(num_unlabeled)
# 반지도 학습 GAN 훈련
batch_size = 64
epochs = 30000

for epoch in range(epochs):
    # 레이블 있는 데이터와 없는 데이터 샘플링
    idx_labeled = np.random.choice(labeled_indices, batch_size // 2, replace=True)
    print(len(idx_labeled))
    # idx_unlabeled = np.random.choice(unlabeled_indices, batch_size // 2, replace=True)
    real_images = [images[idx] for idx in idx_labeled]
    fake_labels = np.zeros((batch_size // 2, 1))
    real_labels = np.ones((64, 1))

    # 판별자 훈련
    print(real_images[0].shape)
    print(real_labels[0].shape)
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(unlabeled_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # 생성자 훈련
    noise = np.random.normal(0, 1, (batch_size // 2, 100))
    g_loss = sgan.train_on_batch(noise, fake_labels)

    # 100번의 에포크마다 생성된 가짜 이미지 저장
    if epoch % 100 == 0:
        r, c = 5, 5  # 저장할 이미지의 행과 열 개수
        noise = np.random.normal(0, 1, (r * c, 100))  # 가짜 이미지를 생성하기 위한 노이즈 생성
        generated_images = generator.predict(noise)  # 생성된 이미지 예측

        # 생성된 이미지를 0-1 범위로 변환
        generated_images = 0.5 * generated_images + 0.5

        # 이미지를 그리드 형태로 배열하여 저장
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(generated_images[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"{output_dir}gan_generated_image_epoch_{epoch}.png")
        plt.close()

    # 1000번의 에포크마다 결과 출력
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, D Accuracy: {100 * d_loss[1]}, G Loss: {g_loss}")
