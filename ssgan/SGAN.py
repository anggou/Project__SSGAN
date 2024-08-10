import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define the number of classes
num_classes = 6  # 10


def build_generator(latent_dim):
    model = keras.Sequential()
    model.add(layers.Dense(128, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    # model.add(layers.Dense(28 * 28 * 3, activation="tanh"))  # Changed to have 3 output units
    model.add(layers.Dense(28 * 28 * 3, activation="sigmoid"))  # Changed to have 3 output units
    model.add(layers.Reshape((28, 28, 3)))  # Reshape to (28, 28, 3) instead of (28, 28, 1)
    return model


def build_discriminator():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same",
                            input_shape=(28, 28, 3)))  # For RGB images, use input_shape=(28, 28, 3)
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def build_classifier():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same",
                            input_shape=(28, 28, 3)))  # For RGB images, use input_shape=(28, 28, 3)
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


# Define the dimensions
latent_dim = 100

# Build and compile the models
generator = build_generator(latent_dim)
discriminator = build_discriminator()
classifier = build_classifier()

# Define the combined SGAN model
discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
classification_output = classifier(generated_image)
gan = keras.Model(gan_input, [gan_output, classification_output])
gan.compile(loss=["binary_crossentropy", "categorical_crossentropy"], optimizer="adam", metrics=['accuracy'])

# Training loop
batch_size = 100
num_epochs = 100
half_batch_size = batch_size // 2

# Set the path to the image samples
image_path = "../SGAN/images/L_1"


# Define the image dimensions
# image_width, image_height = 28, 28


def load_images(path, batch_size):
    images = []
    for filename in os.listdir(path)[:batch_size]:
        image = Image.open(os.path.join(path, filename))
        image = image.resize((image_width, image_height))
        image = np.array(image) / 255.0  # Normalize pixel values
        images.append(image)
    return np.array(images)

def crop_center_and_resize(input_path, output_path, target_width, target_height):
    # 이미지 열기
    image = load_images

    # 이미지 중앙에서 자르기
    left = (image.width - target_width) // 2
    upper = (image.height - target_height) // 2
    right = left + target_width
    lower = upper + target_height

    cropped_image = image.crop((left, upper, right, lower))

    # 이미지 크기 조정
    resized_image = cropped_image.resize((target_width, target_height), Image.ANTIALIAS)

    # 조정된 이미지 저장
    resized_image.save(output_path)

target_width = 820
target_height = 800

for epoch in range(num_epochs):
    for _ in range(batch_size // half_batch_size):  # Run the loop multiple times to cover all batches
        # Generate random noise for the generator
        noise = np.random.normal(0, 1, (half_batch_size, latent_dim))

        # Generate a batch of fake images
        generated_images = generator.predict(noise)
        # Load a batch of real images
        real_images = load_images(image_path, half_batch_size)

        # Concatenate real and fake images
        combined_images = np.concatenate([generated_images, real_images], axis=0)
        # Create labels for the discriminator
        labels = np.concatenate([np.zeros((half_batch_size, 1)), np.ones((half_batch_size, 1))])

        # Compile the discriminator model (You can move this outside the loop as it doesn't change during training)
        discriminator.compile(loss="binary_crossentropy", optimizer="adam")

        # Train the discriminator
        discriminator.trainable = True
        discriminator_loss = discriminator.train_on_batch(combined_images, labels)

        # Train the generator
        discriminator.trainable = False
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        # Generate random labels for the generator (0부터 num_classes - 1까지의 레이블 생성)
        random_array = np.random.randint(0, num_classes, batch_size)

        # Train the generator with the integer labels (not one-hot encoded)
        gan_loss = gan.train_on_batch(noise, [np.ones((batch_size, 1)), random_array])
        discriminator_loss = [discriminator_loss]  # Convert to list
        gan_loss = [gan_loss[0]]  # Convert to list
        # Print the loss metrics
        print(
            f"Epoch: {epoch + 1}/{num_epochs}, Discriminator Loss: {discriminator_loss[0]}, Generator Loss: {gan_loss[0]}")

        # Generate and save sample images
        if epoch % 10 == 0:
            noise = np.random.normal(0, 1, (25, latent_dim))
            generated_images = generator.predict(noise)

            # Rescale images to [0, 1]
            generated_images = 0.5 * generated_images + 0.5

            # Save sample images
            fig, axs = plt.subplots(5, 5)
            count = 0
            for i in range(5):
                for j in range(5):
                    axs[i, j].imshow(generated_images[count, :, :, 0], cmap="gray")
                    axs[i, j].axis("off")
                    count += 1
            plt.savefig(f"generated_images_epoch_{epoch}.png")
            plt.close()

    # Save the generator and discriminator models
    generator.save("generator_model.h5")
    discriminator.save("discriminator_model.h5")
