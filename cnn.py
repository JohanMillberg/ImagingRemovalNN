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

def convolutional_autoencoder():
    inputs = layers.Input(shape=(344, 168, 1))

    p1 = layers.Conv2D(8, (2, 2), activation='relu', padding='same', strides=2)(inputs)
    p2 = layers.Conv2D(16, (2, 2), activation='relu', padding='same', strides=2)(p1)
    p3 = layers.Conv2D(32, (2, 2), activation='relu', padding='same', strides=2)(p2)

    middle = layers.Conv2D(64, (2, 2), padding='same', activation='relu')(p3)

    p4 = layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=2, activation='relu', padding='same')(middle)
    p5 = layers.Conv2DTranspose(16, kernel_size=(2, 2), strides=2, activation='relu', padding='same')(p4)
    p6 = layers.Conv2DTranspose(8, kernel_size=(2, 2), strides=2, activation='relu', padding='same')(p5)

    outputs = layers.Conv2D(1, kernel_size=(1, 1), padding='same')(p6)

    model = tf.keras.Model(inputs, outputs)

    return model


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
    inputs = layers.Input(shape=(344, 168, 1))

    f1, p1 = contracting_layers(inputs, 16, 2, 2) 
    f2, p2 = contracting_layers(p1, 32, 2, 2)
    f3, p3 = contracting_layers(p2, 64, 2, 2)

    middle = layers.Conv2D(128, 2, padding='same', activation='relu')(p3)

    u6 = expanding_layers(middle, f3, 64, 2, 2)
    u7 = expanding_layers(u6, f2, 32, 2, 2)
    u8 = expanding_layers(u7, f1, 16, 2, 2)

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


def get_images(file_name):
    im_indices = get_imaging_indices(25, 81, 512, 175, 350)
    x_image = np.load(f"./images/data/{file_name}").reshape((350, 175))
    y_image = np.load(f"./images/labels/{file_name}")[im_indices].reshape((350, 175))
    x_image = tf.image.resize(preprocess_data(x_image)[tf.newaxis, ..., tf.newaxis], (344, 168))
    y_image = tf.image.resize(preprocess_data(y_image)[tf.newaxis, ..., tf.newaxis], (344, 168))

    return x_image, y_image


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

    x_train_images = tf.image.resize(x_train_images[..., tf.newaxis], (344, 168))
    y_train_images = tf.image.resize(y_train_images[..., tf.newaxis], (344, 168))
    x_test_images = tf.image.resize(x_test_images[..., tf.newaxis], (344, 168))
    y_test_images = tf.image.resize(y_test_images[..., tf.newaxis], (344, 168))

    return x_train_images, y_train_images, x_test_images, y_test_images


def get_imaging_indices(O_x, O_y, image_width, N_x_im, N_y_im):
    im_y_indices = range(O_y, O_y+N_y_im)
    im_x_indices = range(O_x, O_x+N_x_im)
    indices = [y*image_width + x for y in im_y_indices for x in im_x_indices] 

    return indices


def preprocess_data(image_array: np.array):
    return (image_array - np.min(image_array)) / np.ptp(image_array)


def plot_comparison(n_images, imaging_result, reconstructed_images, label_images):

    for i in range(n_images):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plt.gray()

        plot_image(ax1, imaging_result[i], "Imaging algorithm result")

        plot_image(ax2, reconstructed_images[i], "Output of CNN")

        plot_image(ax3, label_images[i], "Actual fracture image")

        plt.savefig(f"images/pngs/im{i}")
        plt.show()

    print("Images saved.")


def plot_image(ax, image, title):
    ax.imshow(tf.squeeze(image))
    ax.set_title(title)
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 


def train_model(x_train, y_train):
    artifact_remover = artifact_remover_unet()
    # artifact_remover = convolutional_autoencoder()
    # loss and optimizer
    optim = keras.optimizers.Adam(learning_rate=0.001)

    artifact_remover.compile(loss=sobel_loss, optimizer=optim)
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
    
    if str(sys.argv[1]) == "load":
        artifact_remover = tf.keras.models.load_model("./saved_model/trained_model.h5", compile=False)
        optim = keras.optimizers.Adam(learning_rate=0.001)
        metrics = ["accuracy"]
        artifact_remover.compile(metrics=metrics, loss=sobel_loss, optimizer=optim)

    else:
        artifact_remover = train_model(x_train, y_train)

    artifact_remover.evaluate(x_test, y_test)

    images_to_decode = 10
    one_frac_x, one_frac_y = get_images("one_frac.npy")

    decoded_images = artifact_remover(x_test[:11])

    print("Average earth mover distance: ", calculate_emd(y_test[:images_to_decode+1,:,:,:], decoded_images))

    plot_comparison(images_to_decode, x_test[:11], decoded_images, y_test[:11])