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

def residual_layer_block(inputs, n_filters, kernel_size, strides=1):
    x = layers.Conv2D(n_filters, kernel_size, strides, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    y = layers.Activation('relu')(x)

    y = layers.Conv2D(n_filters, kernel_size, strides, padding='same', activation='relu')(y) 
    y = layers.Add()([x, y])

    return y


def residual_network():
    inputs = layers.Input(shape=(350, 175, 1))

    x = layers.Conv2D(16, 5, padding='same', strides=5)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(5, padding='same')(x)

    for filters in (16, 32, 64, 128):
        x = residual_layer_block(x, filters, 2)

    x = layers.Conv2DTranspose(16, 5, 5, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(16, 5, 5, padding='same', activation='relu')(x)
    outputs = layers.Conv2D(1, kernel_size=(1, 1), padding='same')(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def convolutional_autoencoder():
    inputs = layers.Input(shape=(344, 168, 1))

    p1 = layers.Conv2D(8, (2, 2), activation='relu', padding='same', strides=1)(inputs)
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
    f = layers.Conv2D(n_filters, kernel_size, padding='same')(x)
    f = layers.BatchNormalization()(f)
    f = layers.Activation('relu')(f)

    f = layers.Conv2D(n_filters, kernel_size, padding='same', activation='relu')(f)
    p = layers.MaxPool2D(downsample_stride)(f)

    return f, p

def expanding_layers(x, copied_features, n_filters, kernel_size, upsample_stride):
    x = layers.Conv2DTranspose(n_filters, kernel_size, upsample_stride, padding='same', activation='relu')(x)
    x = layers.concatenate([x, copied_features])
    x = layers.Conv2D(n_filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.Conv2D(n_filters, kernel_size, padding='same', activation='relu')(x)

    return x


def artifact_remover_unet():
    inputs = layers.Input(shape=(350, 175, 1))

    f1, p1 = contracting_layers(inputs, 16, 5, 5) 

    x = layers.Conv2D(32, 5, padding='same', activation='relu')(p1)
    x = layers.Conv2D(64, 5, padding='same', activation='relu')(x)

    middle = layers.Conv2D(128, 3, padding='same', activation='relu')(x)

    x = layers.Conv2D(64, 5, padding='same', activation='relu')(middle)
    x = layers.Conv2D(32, 5, padding='same', activation='relu')(x)

    u8 = expanding_layers(x, f1, 16, 5, 5)

    outputs = layers.Conv2D(1, 1, padding='same')(u8)
    
    model = tf.keras.Model(inputs, outputs)

    return model


def calculate_emd(target, predicted):

    ws_distances = []
    for i in range(target.shape[0]):
        t_hist, _ = np.histogram(target[i, :, :, :], bins=256, density = True)
        print(t_hist)
        p_hist, _ = np.histogram(predicted[i, :, :, :], bins=256, density = True)

        ws_distances.append(wasserstein_distance(t_hist, p_hist))
    return np.mean(ws_distances)


def sobel_loss(target, predicted):
    sobel_target = tf.image.sobel_edges(target)
    sobel_predicted = tf.image.sobel_edges(predicted)
    
    return tf.reduce_mean(tf.square((sobel_target - sobel_predicted)))


def ssim_loss(target, predicted):
    loss = 1 - tf.reduce_mean(tf.image.ssim(target, predicted, max_val=1.0))
    return loss


def get_images(file_name):
    im_indices = get_imaging_indices(25, 81, 512, 175, 350)
    x_image = np.load(f"./images/data/{file_name}").reshape((350, 175))
    y_image = np.load(f"./images/labels/{file_name}")[im_indices].reshape((350, 175))
    x_image = tf.image.resize(preprocess_data(x_image)[tf.newaxis, ..., tf.newaxis], (344, 168))
    y_image = tf.image.resize(preprocess_data(y_image)[tf.newaxis, ..., tf.newaxis], (344, 168))

    return x_image, y_image


def load_images(image_directory: str,
                n_images: int,
                validation_split: float,
                resize: bool):

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

    x_train_images = x_train_images[..., tf.newaxis]
    y_train_images = y_train_images[..., tf.newaxis]
    x_test_images = x_test_images[..., tf.newaxis]
    y_test_images = y_test_images[..., tf.newaxis]

    if resize:
        x_train_images = tf.image.resize(x_train_images, (344, 168))
        y_train_images = tf.image.resize(y_train_images, (344, 168))
        x_test_images = tf.image.resize(x_test_images, (344, 168))
        y_test_images = tf.image.resize(y_test_images, (344, 168))

    return x_train_images, y_train_images, x_test_images, y_test_images


def get_imaging_indices(O_x, O_y, image_width, N_x_im, N_y_im):
    im_y_indices = range(O_y, O_y+N_y_im)
    im_x_indices = range(O_x, O_x+N_x_im)
    indices = [y*image_width + x for y in im_y_indices for x in im_x_indices] 

    return indices


def preprocess_data(image_array: np.array):
    return (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))


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


def train_model(x_train, y_train, model_name):
    if model_name == "UNet":
        artifact_remover = artifact_remover_unet()
    elif model_name == "ResNet":
        artifact_remover = residual_network()
    elif model_name == "ConvAuto":
        artifact_remover = convolutional_autoencoder()
    else:
        raise NotImplementedError()

    # loss and optimizer
    optim = keras.optimizers.Adam(learning_rate=0.001)
    loss = sobel_loss

    artifact_remover.compile(loss=loss, optimizer=optim)
    artifact_remover.fit(x_train,
            y_train,
            epochs=200,
            shuffle=False,
            batch_size=10,
            verbose=2)
            #validation_data=(x_test, y_test))

    artifact_remover.save(f"./saved_model/{model_name}_trained_model.h5")
    return artifact_remover

if __name__ == "__main__":

    resize = False
    if (sys.argv[3] == "True"):
        resize = True

    model_name = sys.argv[1]

    x_train, y_train, x_test, y_test = load_images("./images", 2050, 0.01, resize)
    
    if str(sys.argv[2]) == "load":
        artifact_remover = tf.keras.models.load_model(f"./saved_model/{model_name}_trained_model.h5", compile=False)

        # set loss and optimizer here
        optim = keras.optimizers.Adam(learning_rate=0.001)
        loss = sobel_loss
        metrics = ["accuracy"]
        artifact_remover.compile(metrics=metrics, loss=loss, optimizer=optim)

    else:
        artifact_remover = train_model(x_train, y_train, model_name)

    artifact_remover.evaluate(x_test, y_test)

    images_to_decode = 10
    one_frac_x, one_frac_y = get_images("one_frac.npy")

    decoded_images = artifact_remover(x_test[:11])

    print("Average earth mover distance: ", calculate_emd(y_test[:images_to_decode+1,:,:,:], decoded_images))

    plot_comparison(images_to_decode, x_test[:11], decoded_images, y_test[:11])