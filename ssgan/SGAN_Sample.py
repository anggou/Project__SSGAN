import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K
import cv2
import os
import pandas as pd #
import time
from PIL import Image
from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Dense,
                                     Dropout, Flatten, Input, Lambda, Reshape)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


output_dir = "../SGAN/fortime/"
class Dataset:
    def __init__(self, num_labeled):

        # 훈련에 사용할 레이블된 샘플 개수
        self.num_labeled = num_labeled #
        num_classes = 8  # 10
        image_folder_labeled = "../SGAN/testify_image/labeled_128/" #400
        image_folder_unlabeled = "../SGAN/testify_image/labeled_128_test/" #4,104 # shuffled_unlabeled_128

        # 이미지와 레이블을 저장할 리스트 초기화
        x = [] # images
        y = [] # labels
        z = []

        # 현재 경로 내의 파일들을 순회하며 이미지 파일을 찾아 리스트에 추가
        for filename in os.listdir(image_folder_labeled):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):  # 지원하는 이미지 확장자 추가
                # 이미지 파일 열기
                image = Image.open(os.path.join(image_folder_labeled, filename)).convert('L')
                x.append(image) #images.append(image)

                # 파일명에서 레이블 정보 추출 (파일명 예시: 'train_00229_002.png')
                label = filename[-5]
                label = int(label)
                label = to_categorical(label, num_classes)
                y.append(label) #labels.append(label)

        for filename in os.listdir(image_folder_unlabeled):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):  # 지원하는 이미지 확장자 추가
                # 이미지 파일 열기
                unlabeled_image = Image.open(os.path.join(image_folder_unlabeled, filename)).convert('L')
                z.append(unlabeled_image)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # MNIST 데이터셋 적재
        self.x_train = x_train #images
        self.y_train = y_train #labels
        self.z_train = z #images
        self.x_test = x_test #images
        self.y_test = y_test

        def preprocess_imgs(x):
            # [0, 255] 사이 흑백 픽셀 값을 [–1, 1] 사이로 변환
            x = [np.array(img) for img in x]
            x = (np.array(x) - 127.5) / 127.5
            # 이미지 크기를 28x28로 조절
            x_resized = [cv2.resize(img, (28, 28)) for img in x]
            # x_resized = [cv2.resize(img, (128, 128)) for img in x]
            # 너비 × 높이 × 채널로 이미지 차원을 확장
            x_resized = np.expand_dims(x_resized, axis=3)
            return x_resized

        def preprocess_labels(y):
            y = np.array(y)
            return y.reshape(-1, 1)

        # 훈련 데이터
        self.x_train = preprocess_imgs(self.x_train)
        self.y_train = preprocess_labels(self.y_train)
        self.z_train = preprocess_imgs(self.z_train)
        self.x_test = preprocess_imgs(self.x_test)
        self.y_test = preprocess_labels(self.y_test)

    def batch_labeled(self, batch_size):
        # 레이블된 이미지와 레이블의 랜덤 배치 만들기
        idx = np.random.randint(0, self.num_labeled, batch_size)
        imgs = self.x_train[idx]
        labels = self.y_train[idx]
        return imgs, labels

    def batch_unlabeled(self, batch_size):
        # 레이블이 없는 이미지의 랜덤 배치 만들기
        idx = np.random.randint(self.num_labeled, self.z_train.shape[0],
                                batch_size)
        imgs = self.z_train[idx]
        return imgs

    def training_set(self):
        x_train = self.x_train[range(self.num_labeled)]
        y_train = self.y_train[range(self.num_labeled)]
        return x_train, y_train

    def test_set(self):
        return self.x_test, self.y_test  ##왜 처리안함

# 사용할 레이블된 샘플 개수 (나머지는 레이블없이 사용합니다)
num_labeled = 60
iterations = 5000

dataset = Dataset(num_labeled) ##

img_rows = 28 #128
img_cols = 28 #128
channels = 1

# 입력 이미지 차원
img_shape = (img_rows, img_cols, channels)

# 생성자의 입력으로 사용할 잡음 벡터의 크기
z_dim = 100 # 100

# 데이터셋에 있는 클래스 개수
num_classes = 8 #10

def build_generator(z_dim):

    model = Sequential()

    # 완전 연결 층을 사용해 입력을 7 × 7 × 256 크기 텐서로 바꿉니다.
    model.add(Dense(256 * 7 * 7, input_dim=z_dim))
    model.add(Reshape((7, 7, 256)))

    # 7 × 7 × 256에서 14 × 14 × 128 텐서로 바꾸는 전치 합성곱 층
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))

    # 배치 정규화
    model.add(BatchNormalization())

    # LeakyReLU 활성화 함수
    model.add(LeakyReLU(alpha=0.01))

    # 14 × 14 × 128에서 14 × 14 × 64 텐서로 바꾸는 전치 합성곱 층
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))

    # 배치 정규화
    model.add(BatchNormalization())

    # LeakyReLU 활성화 함수
    model.add(LeakyReLU(alpha=0.01))

    # 14 × 14 × 64에서 28 × 28 × 1 텐서로 바꾸는 전치 합성곱 층
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))

    # tanh 활성화 함수
    model.add(Activation('tanh'))

    return model

# def build_generator(z_dim):
#     model = Sequential()
#
#     # 완전 연결 층을 사용해 입력을 8 × 8 × 256 크기 텐서로 바꿉니다.
#     model.add(Dense(256 * 8 * 8, input_dim=z_dim))
#     model.add(Reshape((8, 8, 256)))
#
#     # 8 × 8 × 256에서 16 × 16 × 128 텐서로 바꾸는 전치 합성곱 층
#     model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
#
#     # 배치 정규화
#     model.add(BatchNormalization())
#
#     # LeakyReLU 활성화 함수
#     model.add(LeakyReLU(alpha=0.01))
#
#     # 16 × 16 × 128에서 32 × 32 × 64 텐서로 바꾸는 전치 합성곱 층
#     model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
#
#     # 배치 정규화
#     model.add(BatchNormalization())
#
#     # LeakyReLU 활성화 함수
#     model.add(LeakyReLU(alpha=0.01))
#
#     # 32 × 32 × 64에서 64 × 64 × 32 텐서로 바꾸는 전치 합성곱 층
#     model.add(Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'))
#
#     # 배치 정규화
#     model.add(BatchNormalization())
#
#     # LeakyReLU 활성화 함수
#     model.add(LeakyReLU(alpha=0.01))
#
#     # 64 × 64 × 32에서 128 × 128 × 1 텐서로 바꾸는 전치 합성곱 층
#     model.add(Conv2DTranspose(1, kernel_size=4, strides=2, padding='same'))
#
#     # tanh 활성화 함수
#     model.add(Activation('tanh'))
#
#     return model

def build_discriminator_net(img_shape):
    model = Sequential()

    # 28 × 28 × 1에서 14 × 14 × 32 텐서로 바꾸는 합성곱 층
    model.add(
        Conv2D(32,
               kernel_size=3,
               strides=4, #2
               input_shape=img_shape,
               padding='same'))

    # LeakyReLU 활성화 함수
    model.add(LeakyReLU(alpha=0.01))

    # 14 × 14 × 32에서 7 × 7 × 64 텐서로 바꾸는 합성곱 층
    model.add(
        Conv2D(64,
               kernel_size=3,
               strides=2,
               padding='same'))

    # LeakyReLU 활성화 함수
    model.add(LeakyReLU(alpha=0.01))

    # 7 × 7 × 64에서 3 × 3 × 128 텐서로 바꾸는 합성곱 층
    model.add(
        Conv2D(128,
               kernel_size=3,
               strides=2,
               padding='same'))

    # LeakyReLU 활성화 함수
    model.add(LeakyReLU(alpha=0.01))

    # 드롭아웃
    model.add(Dropout(0.5))

    # 텐서 펼치기
    model.add(Flatten())

    # num_classes 개의 뉴런을 가진 완전 연결 층
    model.add(Dense(num_classes))

    return model

# def build_discriminator_net(img_shape):
#     model = Sequential()
#
#     # 128 × 128 × 1에서 64 × 64 × 64 텐서로 바꾸는 합성곱 층
#     model.add(
#         Conv2D(64,
#                kernel_size=4,
#                strides=2,
#                input_shape=img_shape,
#                padding='same'))
#
#     # LeakyReLU 활성화 함수
#     model.add(LeakyReLU(alpha=0.01))
#
#     # 64 × 64 × 64에서 32 × 32 × 128 텐서로 바꾸는 합성곱 층
#     model.add(
#         Conv2D(128,
#                kernel_size=4,
#                strides=2,
#                padding='same'))
#
#     # LeakyReLU 활성화 함수
#     model.add(LeakyReLU(alpha=0.01))
#
#     # 32 × 32 × 128에서 16 × 16 × 256 텐서로 바꾸는 합성곱 층
#     model.add(
#         Conv2D(256,
#                kernel_size=4,
#                strides=2,
#                padding='same'))
#
#     # LeakyReLU 활성화 함수
#     model.add(LeakyReLU(alpha=0.01))
#
#     # 드롭아웃
#     model.add(Dropout(0.5))
#
#     # 텐서 펼치기
#     model.add(Flatten())
#
#     # num_classes 개의 뉴런을 가진 완전 연결 층
#     model.add(Dense(num_classes))
#
#     return model


def build_discriminator_supervised(discriminator_net):

    model = Sequential()

    model.add(discriminator_net)

    # 진짜 클래스에 대한 예측 확률을 출력하는 소프트맥스 활성화 함수
    model.add(Activation('softmax'))

    return model

def build_discriminator_unsupervised(discriminator_net):

    model = Sequential()

    model.add(discriminator_net)

    def predict(x):
        # 진짜 클래스에 대한 확률 분포를 진짜 대 가짜의 이진 확률로 변환합니다. ###
        prediction = 1.0 - (1.0 /
                            (K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0))
        return prediction

    # 앞서 정의한 진짜 대 가짜 확률을 출력하는 뉴런
    model.add(Lambda(predict))

    return model

def build_gan(generator, discriminator):

    model = Sequential()

    # 생성자와 판별자 모델을 연결하기
    model.add(generator)
    model.add(discriminator)

    return model

def save_generated_images(generator, iteration, examples=10, dim=(1, 10), figsize=(10, 1)):
    z = np.random.normal(0, 1, (examples, z_dim))
    gen_imgs = generator.predict(z)
    # gen_imgs = 0.5 * gen_imgs + 0.5  # 이미지를 [0, 1] 범위로 변환합니다.
    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
    plt.subplots_adjust(wspace=0, hspace=0)

    for i in range(dim[0] * dim[1]):
        ax = axs[i]
        ax.imshow(gen_imgs[i], cmap='gray')
        ax.axis('off')

    fig.savefig(f"{output_dir}gan_generated_image_epoch_{iteration}.png")
    plt.close()

# 판별자 기반 모델: 이 층들은 지도 학습 훈련과 비지도 학습 훈련에 공유됩니다.

discriminator_net = build_discriminator_net(img_shape)

# 지도 학습 훈련을 위해 판별자를 만들고 컴파일합니다.
discriminator_supervised = build_discriminator_supervised(discriminator_net)
discriminator_supervised.compile(loss='categorical_crossentropy',
                                 metrics=['accuracy'],
                                 optimizer=Adam(learning_rate=0.0003))

# 비지도 학습 훈련을 위해 판별자를 만들고 컴파일합니다.
discriminator_unsupervised = build_discriminator_unsupervised(discriminator_net)
discriminator_unsupervised.compile(loss='binary_crossentropy',
                                   optimizer=Adam())

# 생성자를 만듭니다.
generator = build_generator(z_dim)

# 생성자 훈련을 위해 판별자의 모델 파라미터를 동결합니다.
discriminator_unsupervised.trainable = False

# 생성자를 훈련하기 위해 고정된 판별자로 GAN 모델을 만들고 컴파일합니다.
gan = build_gan(generator, discriminator_unsupervised)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

supervised_losses = []
unsupervised_losses = [] #
generator_losses = [] #
iteration_checkpoints = []

model_path = "../SGAN/fortime/"  # 모델 저장 경로


def train(iterations, batch_size, sample_interval):

    # 진짜 이미지의 레이블: 모두 1
    real = np.ones((batch_size, 1))

    # 가짜 이미지의 레이블: 모두 0
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):
        # -------------------------
        #  판별자 훈련
        # -------------------------
        # 레이블된 샘플을 가져옵니다.
        imgs, labels = dataset.batch_labeled(batch_size)

        # 레이블을 원-핫 인코딩합니다.
        labels = to_categorical(labels, num_classes=num_classes)

        # 레이블이 없는 샘플을 가져옵니다.
        imgs_unlabeled = dataset.batch_unlabeled(batch_size)

        # 가짜 이미지의 배치를 생성합니다.
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        # 레이블된 진짜 샘플에서 훈련합니다.
        d_loss_supervised, accuracy = discriminator_supervised.train_on_batch(imgs, labels)

        # 레이블이 없는 진짜 discriminator_unsupervised.샘플에서 훈련합니다.
        d_loss_real = discriminator_unsupervised.train_on_batch(imgs_unlabeled, real)

        # 가짜 샘플에서 훈련합니다.
        d_loss_fake = discriminator_unsupervised.train_on_batch(gen_imgs, fake)

        d_loss_unsupervised = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 생성자를 훈련합니다.
        g_loss = gan.train_on_batch(z, np.ones((batch_size, 1)))

        if (iteration + 1) % sample_interval == 0:

            # 훈련이 끝난 후 그래프를 그리기 위해 판별자의 지도 학습 분류 손실을 기록합니다.
            supervised_losses.append(d_loss_supervised)
            unsupervised_losses.append(d_loss_unsupervised) #
            generator_losses.append(g_loss) #
            iteration_checkpoints.append(iteration + 1)

            # 훈련 과정을 출력합니다.
            print(
                "%d [D loss supervised: %.4f, acc.: %.2f%%] [D loss unsupervised: %.4f] [G loss: %f]"
                % (iteration + 1, d_loss_supervised, 100 * accuracy,
                   d_loss_unsupervised, g_loss))

        if (iteration + 1) % (sample_interval * 2) == 0:
            save_generated_images(generator, iteration + 1)

        if (iteration + 1) % 100 == 0:
            model_filename = f"sgan_model_epoch_{iteration + 1:04d}_128.keras"  # 모델 파일 이름
            # 모델을 저장
            gan.save(os.path.join(model_path, model_filename))

        # 이 부분에 데이터프레임을 Excel 파일로 내보내는 코드를 추가합니다.
        if (iteration + 1) == iterations:
            data = {
                'Iteration': iteration_checkpoints,
                'Generator Loss': generator_losses,
                'Discriminator Supervised Loss': supervised_losses,
                'Discriminator Unsupervised Loss': unsupervised_losses
            }
            df = pd.DataFrame(data)

            # 저장할 Excel 파일 경로와 파일명을 따로 지정합니다.
            directory_path = model_path
            file_name = "classification_results.xlsx"

            excel_path = directory_path + file_name

            # 데이터프레임을 Excel 파일로 내보냅니다.
            df.to_excel(excel_path, index=False, sheet_name='Training Results')

# 하이퍼파라미터를 셋팅합니다.
# iterations = 30 #8000
batch_size = 32
sample_interval = 10

# 지정한 반복 횟수 동안 SGAN을 훈련합니다.

start_time = time.time()

train(iterations, batch_size, sample_interval)

end_time = time.time()
training_duration = end_time - start_time
print(f"학습에 걸린 시간은 {training_duration:.2f}초입니다")
losses = np.array(supervised_losses)

# 판별자의 지도 학습 손실을 그립니다.
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, losses, label="Discriminator loss")

plt.xticks(iteration_checkpoints, rotation=90)

plt.title("Discriminator – Supervised Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()

x, y = dataset.training_set()
y = to_categorical(y, num_classes=num_classes)

# 훈련 세트에서 분류 정확도 계산
_, accuracy = discriminator_supervised.evaluate(x, y)
print("Training Accuracy: %.2f%%" % (100 * accuracy))
x, y = dataset.test_set()
y = to_categorical(y, num_classes=num_classes)

# 테스트 세트에서 분류 정확도 계산
_, accuracy = discriminator_supervised.evaluate(x, y)
print("Test Accuracy: %.2f%%" % (100 * accuracy))

#
# # SGAN 판별자와 같은 네트워크 구조를 가진 지도 학습 분류기
# mnist_classifier = build_discriminator_supervised(build_discriminator_net(img_shape))
# mnist_classifier.compile(loss='categorical_crossentropy',
#                          metrics=['accuracy'],
#                          optimizer=Adam())
# imgs, labels = dataset.training_set()
#
# # 레이블을 원-핫 인코딩합니다.
# labels = to_categorical(labels, num_classes=num_classes)
#
# # 분류기를 훈련합니다.
# training = mnist_classifier.fit(x=imgs,
#                                 y=labels,
#                                 batch_size=batch_size,
#                                 epochs=30,
#                                 verbose=1)
# losses = training.history['loss']
# accuracies = training.history['accuracy']
# # 분류 손실을 그립니다
# plt.figure(figsize=(10, 5))
# plt.plot(np.array(losses), label="Loss")
# plt.title("Classification Loss")
# plt.legend()
# plt.show()
# # 분류 정확도를 그립니다.
# plt.figure(figsize=(10, 5))
# plt.plot(np.array(accuracies), label="Accuracy")
# plt.title("Classification Accuracy")
# plt.legend()
# plt.show()
# x, y = dataset.training_set()
# y = to_categorical(y, num_classes=num_classes)
#
# # 훈련 세트에 대한 분류 정확도를 계산합니다.
# _, accuracy = mnist_classifier.evaluate(x, y)
# print("Training Accuracy: %.2f%%" % (100 * accuracy))
# x, y = dataset.test_set()
# y = to_categorical(y, num_classes=num_classes)
#
# # 테스트 세트에 대한 분류 정확도를 계산합니다.
# _, accuracy = mnist_classifier.evaluate(x, y)
# print("Test Accuracy: %.2f%%" % (100 * accuracy))

# # 이 부분에 데이터프레임을 Excel 파일로 내보내는 코드를 추가합니다.
# data = {
#     'Iteration': iteration_checkpoints,
#     'Generator Loss': generator_losses,
#     'Discriminator Supervised Loss': supervised_losses,
#     'Discriminator Unsupervised Loss': unsupervised_losses
# }
#
# # 데이터프레임 생성
# df = pd.DataFrame(data)
#
# # Excel 파일로 저장
# output_file = os.path.join(output_dir, "classification_results.xlsx")
# df.to_excel(output_file, index=False, sheet_name='Training Results')
