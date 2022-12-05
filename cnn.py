import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from scipy.signal import convolve2d as conv2
from scipy.stats import truncnorm
from PIL import Image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# tf.config.run_functions_eagerly(True)

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
            layers.Input(shape=(350, 175, 1)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(4, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(1, (2, 2), activation='relu', padding='same')
        ])
        print(self.model.summary())

    def call(self, x):
        return self.model(x)



def dual_convolutional_block(x, kernel_size, n_filters):
    x = layers.Conv2D(n_filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.Conv2D(n_filters, kernel_size, padding='same', activation='relu')(x)

    return x

def downsample_block(x, n_filters, kernel_size, downsample_stride):
    f = dual_convolutional_block(x, kernel_size, n_filters)
    p = layers.MaxPool2D(downsample_stride)(f)
    p = layers.Dropout(0.2)(p)

    return f, p

def upsample_block(x, conv_features, n_filters, kernel_size, upsample_stride):
    x = layers.Conv2DTranspose(n_filters, kernel_size, upsample_stride, padding='same')(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(0.2)(x)
    x = dual_convolutional_block(x, kernel_size, n_filters)

    return x

def artifact_remover_unet():
    inputs = layers.Input(shape=(350, 175, 1))

    f1, p1 = downsample_block(inputs, 8, 5, 5) 
    f2, p2 = downsample_block(p1, 16, 5, 5)
    f3, p3 = downsample_block(p2, 32, 7, 7)

    latent = dual_convolutional_block(p3, 5, 32)

    u6 = upsample_block(latent, f3, 32, 7, 7)
    u7 = upsample_block(u6, f2, 16, 5, 5)
    u8 = upsample_block(u7, f1, 8, 5, 5)

    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(u8)
    
    model = tf.keras.Model(inputs, outputs)

    return model

def mse_sobel_loss(target, predicted):
    sobel_target = tf.image.sobel_edges(target)
    sobel_predicted = tf.image.sobel_edges(predicted)
    mse = tf.keras.losses.MeanSquaredError()
    
    gamma = 1.0 # weight of sobel loss

    return tf.math.add(
            tf.math.scalar_mul(gamma, K.mean(K.square(sobel_target - sobel_predicted))),
            tf.math.scalar_mul((1-gamma), mse(target, predicted))
    )

def load_images(image_directory: str,
                n_images: int,
                validation_split: float):

    x_img_array_list = []
    y_img_array_list = []

    im_indices = get_imaging_indices(25, 81, 512, 175, 350)

    for i in range(n_images):
        x_img_array = np.load(f"{image_directory}/data/im{i}.npy").reshape((350, 175))
        x_img_array_list.append(preprocess_data(x_img_array))

        y_img_array = np.load(f"{image_directory}/labels/im{i}.npy")[im_indices]
        y_img_array = y_img_array.reshape((350, 175))
        y_img_array_list.append(preprocess_data(y_img_array))


    x_image_tensor = np.stack(x_img_array_list, axis=0)
    y_image_tensor = np.stack(y_img_array_list, axis=0)

    training_index = int(x_image_tensor.shape[0]*(1-validation_split))
    x_train_images = x_image_tensor[:training_index, :, :]
    y_train_images = y_image_tensor[:training_index, :, :]

    x_test_images = x_image_tensor[training_index:, :, :]
    y_test_images = y_image_tensor[training_index:, :, :]

    return x_train_images, y_train_images, x_test_images, y_test_images

def get_imaging_indices(O_x, O_y, image_width, N_x_im, N_y_im):
    im_y_indices = range(O_y, O_y+N_y_im)
    im_x_indices = range(O_x, O_x+N_x_im)
    indices = [y*image_width + x for y in im_y_indices for x in im_x_indices] 

    return indices

def preprocess_data(image_array: np.array):
    return (image_array - np.min(image_array)) / np.ptp(image_array)

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
        gauss_noise[gauss_noise < 0] = 0
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

    plt.show()

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_images("./images", 200, 0.2)

    """
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    x_train, x_test = preprocess_data(x_train, x_test)
    """

    x_train = x_train[..., tf.newaxis]
    y_train = y_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    y_test = y_test[..., tf.newaxis]

    artifact_remover = artifact_remover_unet()
    # loss and optimizer
    loss = keras.losses.MeanSquaredError()
    optim = keras.optimizers.Adam(learning_rate=0.001)
    metrics = ["accuracy"]

    artifact_remover.compile(metrics=metrics, loss=mse_sobel_loss, optimizer=optim)
    artifact_remover.fit(x_train,
            y_train,
            epochs=10,
            shuffle=False,
            batch_size=10,
            verbose=2,
            validation_data=(x_test, y_test))

    decoded_images = artifact_remover(x_test)
    plot_comparison(3, x_test, decoded_images)