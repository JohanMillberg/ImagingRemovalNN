import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from scipy.signal import convolve2d as conv2

import os
os.environ['TF_CPP_MIN_LOG_LEBEL'] = '3'


# model
class ArtifactRemover(keras.Model):
    def __init__(self):
        super().__init__()

        self.encoder = keras.models.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)
        ])

        self.decoder = keras.models.Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



def preprocess_data(train_images: np.array, test_images: np.array):
    train_normalized = train_images / 255.0
    test_normalized = test_images / 255.0
    return train_normalized, test_normalized

def convolve_images(images):
    convolved = images.copy()
    psf = np.ones((2,2)) / 25
    n_images = convolved.shape[0]

    for i in range(n_images):
        convolved[i] = conv2(convolved[i], psf, 'same')

    convolved = convolved[..., tf.newaxis]

    return convolved


(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train, x_test = preprocess_data(x_train, x_test)

x_train_convolved = convolve_images(x_train)
x_test_convolved = convolve_images(x_test)

"""
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
          batch_size=10,
          verbose=2,
          validation_data=(x_test_convolved, x_test))

encoded_images = artifact_remover.encoder(x_test_convolved).numpy()
decoded_images = artifact_remover.decoder(encoded_images)
"""
model = keras.models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(18, (4, 4), activation='relu', padding='same'),

])

n = 3 
plt.figure(figsize=(20, 7))
plt.gray()
for i in range(n): 
  # display convoluted originals 
  bx = plt.subplot(3, n, i + 1) 
  plt.title("convoluted originals") 
  plt.imshow(tf.squeeze(x_test_convolved[i])) 
  bx.get_xaxis().set_visible(False) 
  bx.get_yaxis().set_visible(False) 
  
  # display reconstruction 
  cx = plt.subplot(3, n, i + n + 1) 
  plt.title("reconstructed") 
  plt.imshow(tf.squeeze(decoded_images[i])) 
  cx.get_xaxis().set_visible(False) 
  cx.get_yaxis().set_visible(False) 
  
  # display original 
  ax = plt.subplot(3, n, i + 2*n + 1) 
  plt.title("original") 
  plt.imshow(tf.squeeze(x_test[i])) 
  ax.get_xaxis().set_visible(False) 
  ax.get_yaxis().set_visible(False) 

plt.show()