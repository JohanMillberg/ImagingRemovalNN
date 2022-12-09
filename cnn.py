import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_probability import stats
from scipy.signal import convolve2d as conv2
from scipy.stats import wasserstein_distance;
import sys, getopt
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


def contracting_layers(x, n_filters, kernel_size, downsample_stride):
    f = layers.Conv2D(n_filters, kernel_size, padding='same', activation='relu')(x)
    f = layers.Conv2D(n_filters, kernel_size, padding='same', activation='relu')(f)
    p = layers.MaxPool2D(downsample_stride)(f)

    return f, p

def expanding_layers(x, copied_features, n_filters, kernel_size, upsample_stride):
    x = layers.Conv2DTranspose(n_filters, kernel_size, upsample_stride, padding='same')(x)
    x = layers.concatenate([x, copied_features])
    x = layers.Conv2D(n_filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.Conv2D(n_filters, kernel_size, padding='same', activation='relu')(x)

    return x


def artifact_remover_unet():
    inputs = layers.Input(shape=(350, 175, 1))

    f1, p1 = contracting_layers(inputs, 16, 5, 5) 
    f2, p2 = contracting_layers(p1, 32, 5, 5)
    f3, p3 = contracting_layers(p2, 64, 7, 7)

    middle = layers.Conv2D(128, 5, padding='same', activation='relu')(p3)

    u6 = expanding_layers(middle, f3, 64, 7, 7)
    u7 = expanding_layers(u6, f2, 32, 5, 5)
    u8 = expanding_layers(u7, f1, 16, 5, 5)

    outputs = layers.Conv2D(1, 1, padding='same')(u8)
    
    model = tf.keras.Model(inputs, outputs)

    return model


def calculate_emd(target, predicted):

    ws_distances = []
    for i in range(target.shape[0]):
        t_hist, _ = np.histogram(target[i, :, :, :], bins=256, density = True)
        p_hist, _ = np.histogram(predicted[i, :, :, :], bins=256, density = True)

        ws_distances.append(wasserstein_distance(t_hist, p_hist))
    return np.mean(ws_distances)


def sobel_loss(target, predicted):
    sobel_target = tf.image.sobel_edges(target)
    sobel_predicted = tf.image.sobel_edges(predicted)
    
    return K.mean(K.square(sobel_target - sobel_predicted))


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


def plot_comparison(n_images, imaging_result, reconstructed_images, label_images):

    for i in range(n_images):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plt.gray()

        plot_image(ax1, imaging_result[i], "Imaging algorithm result")

        plot_image(ax2, reconstructed_images[i], "Output of CNN")

        plot_image(ax3, label_images[i], "Actual fracture image")

        plt.savefig(f"images/pngs/im{i}")

    print("Images saved.")


def plot_image(ax, image, title):
    ax.imshow(tf.squeeze(image))
    ax.set_title(title)
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 


def train_model(x_train, y_train):
    artifact_remover = artifact_remover_unet()
    # loss and optimizer
    optim = keras.optimizers.Adam(learning_rate=0.001)
    metrics = ["accuracy"]

    artifact_remover.compile(metrics=metrics, loss=sobel_loss, optimizer=optim)
    artifact_remover.fit(x_train,
            y_train,
            epochs=200,
            shuffle=False,
            batch_size=10,
            verbose=2)
            #validation_data=(x_test, y_test))

    artifact_remover.save("./saved_model/trained_model.h5")
    return artifact_remover

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_images("./images", 2050, 0.01)
    x_train = x_train[..., tf.newaxis]
    y_train = y_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    y_test = y_test[..., tf.newaxis]
    
    if str(sys.argv[1]) == "load":
        artifact_remover = tf.keras.models.load_model("./saved_model/trained_model.h5", compile=False)
        optim = keras.optimizers.Adam(learning_rate=0.001)
        metrics = ["accuracy"]
        artifact_remover.compile(metrics=metrics, loss=sobel_loss, optimizer=optim)

    else:
        artifact_remover = train_model(x_train, y_train)

    artifact_remover.evaluate(x_test, y_test)

    images_to_decode = 10

    decoded_images = artifact_remover(x_test[:11])

    print("Average earth mover distance: ", calculate_emd(y_test[:images_to_decode+1,:,:,:], decoded_images))

    plot_comparison(images_to_decode, x_test[:images_to_decode+1], decoded_images[:images_to_decode+1], y_test[:images_to_decode+1])