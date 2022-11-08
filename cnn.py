import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from scipy.signal import convolve2d as conv2
from scipy.stats import truncnorm
from PIL import Image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# model
class ArtifactRemover(keras.Model):
    def __init__(self):
        super().__init__()

        self.encoder = keras.models.Sequential([
            layers.Input(shape=(150, 300, 1)),
            layers.Conv2D(16, (15, 15), activation='relu', padding='same', strides=3),
            layers.Conv2D(8, (6, 6), activation='relu', padding='same', strides=2)
        ])

        self.decoder = keras.models.Sequential([
            layers.Conv2DTranspose(8, kernel_size=6, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=15, strides=3, activation='relu', padding='same'),
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
            layers.Input(shape=(150, 300, 1)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(4, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(1, (2, 2), activation='relu', padding='same')
        ])
        print(self.model.summary())

    def call(self, x):
        return self.model(x)



def dual_convolutional_block(x, n_filters):
    x = layers.Conv2D(n_filters, 6, padding='same', activation='relu')
    x = layers.Conv2D(n_filters, 6, padding='same', activation='relu')

    return x

def downsample_block(x, n_filters):
    f = dual_convolutional_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.2)(p)

    return f, p

def upsample_block(x, conv_features, n_filters):
    x = layers.Conv2DTranspose(n_filters, 6, 2, padding='same')(x)
    x = layers.Concatenate([x, conv_features])
    x = layers.Dropout(0.2)(x)
    x = dual_convolutional_block(x, n_filters)

    return x

def artifact_remover_unet(x):
    inputs = layers.Input(shape=(150, 150, 1))

    f1, p1 = downsample_block(inputs, 64) 

    f2, p2 = downsample_block(p1, 128)

    f3, p3 = downsample_block(p2, 256)

    f4, p4 = downsample_block(p3, 512)

    latent = dual_convolutional_block(p4, 1024)

    u6 = upsample_block(latent, f4, 512)

    u7 = upsample_block(u6, f3, 256)

    u8 = upsample_block(u7, f2, 128)

    u9 = upsample_block(u8, f1, 64)

    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(u9)
    
    model = tf.keras.Model(inputs, outputs)


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

def add_noise(images: np.array, mean_noise: float, std_dev_noise: float):
    
    noisy_images = images.copy()
    n_images = noisy_images.shape[0]

    for i in range(n_images):
        gauss_noise = np.random.normal(loc=mean_noise,
                                    scale=std_dev_noise,
                                    size=noisy_images[i].size
                                    )

        gauss_noise = gauss_noise.reshape(*(noisy_images[i]).shape)
        noisy_images[i] = np.add(noisy_images[i], gauss_noise)

    return noisy_images

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

if __name__ == "__main__":
    x_train, x_test = load_images("images/fractured", 150, 300, 1000, 0.2)

    """
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    x_train, x_test = preprocess_data(x_train, x_test)
    """
    x_train_convolved = convolve_images(x_train)
    # x_train_convolved = add_noise(x_train_convolved, 1, 0.1)
    x_test_convolved = convolve_images(x_test)
    # x_test_convolved = add_noise(x_test_convolved, 1, 0.1)
    
    x_train_convolved, x_test_convolved = preprocess_data(x_train_convolved, x_test_convolved)

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    x_train, x_test = preprocess_data(x_train, x_test)

    artifact_remover = ArtifactRemover()
    # loss and optimizer
    loss = keras.losses.MeanSquaredError()
    optim = keras.optimizers.Adam(learning_rate=0.001)
    metrics = ["accuracy"]

    artifact_remover.compile(metrics=metrics, loss=loss, optimizer=optim)
    artifact_remover.fit(x_train_convolved,
            x_train,
            epochs=10,
            shuffle=False,
            batch_size=50,
            verbose=2,
            validation_data=(x_test_convolved, x_test))

    decoded_images = artifact_remover(x_test_convolved)
    plt.gray()
    plt.imshow(tf.squeeze(decoded_images[0]))
    plt.show()
    plot_comparison(3, x_test_convolved, decoded_images)