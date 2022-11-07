import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from scipy.signal import convolve2d as conv2
from PIL import Image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# model
class ArtifactRemover(keras.Model):
    def __init__(self):
        super().__init__()

        self.encoder = keras.models.Sequential([
            layers.Input(shape=(150, 300, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=1),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=1)
        ])

        self.decoder = keras.models.Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=1, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=1, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ArtifactRemoverV2(keras.Model):
    def __init__(self):
        super().__init__()

        self.model = keras.models.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(4, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(1, (2, 2), activation='relu', padding='same')
        ])
        print(self.model.summary())

    def call(self, x):
        return self.model(x)


def load_images(image_directory: str,
                image_height: int,
                image_width: int,
                n_images: int,
                validation_split: float):

    img_array_list = []

    for i in range(n_images):
        img_array = np.array(Image.open(f"{image_directory}/im{i}.jpg").convert('L'))
        img_array_list.append(img_array)


    image_tensor = np.stack(img_array_list, axis=0)
    training_index = int(image_tensor.shape[0]*(1-validation_split))
    train_images = image_tensor[:training_index, :, :]
    test_images = image_tensor[training_index:, :, :]

    train_images, test_images = preprocess_data(train_images, test_images)

    return train_images, test_images


def preprocess_data(train_images: np.array, test_images: np.array):
    train_normalized = train_images.astype(float) / np.max(train_images)
    test_normalized = test_images.astype(float) / np.max(test_images)

    return train_normalized, test_normalized

def convolve_images(images):
    convolved = images.copy()
    psf = np.array([[0, 1, 2, 1, 0], [1, 2, 3, 2, 1], [2, 3, 4, 3, 2],
                    [1, 2, 3, 2, 1], [0, 1, 2, 1, 0]], dtype='float64')
    psf *= 1 / np.sum(psf)
    n_images = convolved.shape[0]

    for i in range(n_images):
        convolved[i] = conv2(convolved[i], psf, 'same')

    convolved = convolved[..., tf.newaxis]

    return convolved


def plot_comparison(n_images, convolved_images, reconstructed_images):
 
    n = 3 
    plt.figure(figsize=(20, 7))
    plt.gray()

    for i in range(n): 
        # display convoluted originals 
        bx = plt.subplot(3, n, i + 1) 
        plt.title("convoluted originals") 
        plt.imshow(tf.squeeze(convolved_images[i])) 
        bx.get_xaxis().set_visible(False) 
        bx.get_yaxis().set_visible(False) 
        
        # display reconstruction 
        cx = plt.subplot(3, n, i + n + 1) 
        plt.title("reconstructed") 
        plt.imshow(tf.squeeze(reconstructed_images[i])) 
        cx.get_xaxis().set_visible(False) 
        cx.get_yaxis().set_visible(False) 
        
        # display original 
        ax = plt.subplot(3, n, i + 2*n + 1) 
        plt.title("original") 
        plt.imshow(tf.squeeze(x_test[i])) 
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False) 

    plt.show()


x_train, x_test = load_images("images/fractured", 150, 300, 1000, 0.2)

"""
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train, x_test = preprocess_data(x_train, x_test)
"""
x_train_convolved = convolve_images(x_train)
x_test_convolved = convolve_images(x_test)

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

artifact_remover = ArtifactRemover()
# loss and optimizer
loss = keras.losses.MeanSquaredError()
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

artifact_remover.compile(metrics=metrics, loss=loss, optimizer=optim)
artifact_remover.fit(x_train_convolved,
          x_train,
          epochs=10,
          shuffle=True,
          batch_size=100,
          verbose=2,
          validation_data=(x_test_convolved, x_test))

decoded_images = artifact_remover(x_test_convolved)
plot_comparison(3, x_test_convolved, decoded_images)