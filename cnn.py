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
    x = layers.Conv2D(n_filters, kernel_size, strides, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    y = layers.Conv2D(n_filters, kernel_size, strides, padding='same', activation='relu')(x) 
    y = layers.BatchNormalization()(y)
    y = layers.Add()([x, y])

    return y


def residual_network():
    inputs = layers.Input(shape=(350, 175, 1))

    x = layers.Conv2D(16, 5, padding='same', strides=5, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    for filters in (16, 32, 64):
        x = residual_layer_block(x, filters, 3)

    x = layers.Conv2DTranspose(128, 5, 5, padding='same', activation='relu')(x)
    outputs = layers.Conv2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def convolutional_autoencoder():
    inputs = layers.Input(shape=(344, 168, 1))

    p1 = layers.Conv2D(16, (2, 2), activation='relu', padding='same', strides=2)(inputs)
    p2 = layers.Conv2D(32, (2, 2), activation='relu', padding='same', strides=2)(p1)

    middle = layers.Conv2D(64, (2, 2), padding='same', activation='relu')(p2)

    p3 = layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=2, activation='relu', padding='same')(middle)
    p4 = layers.Conv2DTranspose(16, kernel_size=(2, 2), strides=2, activation='relu', padding='same')(p3)

    outputs = layers.Conv2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid')(p4)

    model = tf.keras.Model(inputs, outputs)

    return model


def contracting_layers(x, n_filters, kernel_size, downsample_stride):
    f = layers.Conv2D(n_filters, kernel_size, padding='same', activation='relu')(x)
    f = layers.BatchNormalization()(f)

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

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(p1)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    middle = layers.Conv2D(128, 3, padding='same', activation='relu')(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(middle)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)

    u8 = expanding_layers(x, f1, 16, 5, 5)

    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(u8)
    
    model = tf.keras.Model(inputs, outputs)

    return model


def calculate_emd(target, predicted):

    ws_distances = []
    for i in range(target.shape[0]):
        t_hist, _ = np.histogram(target[i, :, :, :], bins=256, density = True)
        p_hist, _ = np.histogram(predicted[i, :, :, :], bins=256, density = True)

        ws_distances.append(wasserstein_distance(t_hist, p_hist))
    return np.mean(ws_distances)


def calculate_mse(target, predicted):
    mse_vals = []
    mse = tf.keras.losses.MeanSquaredError()
    for t, p in list(zip(target, predicted)):
        mse_vals.append(mse(t, p))
    
    return np.mean(mse_vals)


def calculate_psnr(target, predicted):
    psnr_vals = []
    for t, p in list(zip(target, predicted)):
        psnr_vals.append(tf.image.psnr(t, p, max_val=1.0))

    return np.mean(psnr_vals)


def calculate_ssim(target, predicted):
    ssim_vals = []
    for t, p in list(zip(target, predicted)):
        ssim_vals.append(tf.reduce_mean(tf.image.ssim(tf.cast(t, tf.float64), tf.cast(p, tf.float64), max_val=1.0)))

    return np.mean(ssim_vals)
    

def sobel_loss(target, predicted):
    sobel_target = tf.image.sobel_edges(target)
    sobel_predicted = tf.image.sobel_edges(predicted)
    
    return tf.reduce_mean(tf.square((sobel_target - sobel_predicted)))


def ssim_loss(target, predicted):
    loss = 1 - tf.reduce_mean(tf.image.ssim(target, predicted, max_val=1.0))
    return loss


def get_images(file_name, resize):
    im_indices = get_imaging_indices(25, 81, 512, 175, 350)
    x_image = np.load(f"./images/data/{file_name}").reshape((350, 175))
    y_image = np.load(f"./images/labels/{file_name}")[im_indices].reshape((350, 175))

    x_image = preprocess_data(x_image)[..., tf.newaxis]
    y_image = preprocess_data(y_image)[..., tf.newaxis]

    if resize:
        x_image = tf.image.resize(x_image, (344, 168))
        y_image = tf.image.resize(y_image, (344, 168))

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


def plot_comparison(n_images,
                    imaging_result,
                    reconstructed_images,
                    label_images,
                    model_name,
                    loss_name,
                    start_index):
    
    save_path = f"images/pngs/{model_name}_{loss_name}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(n_images):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plt.gray()

        plot_image(ax1, imaging_result[i], "Imaging algorithm result")

        plot_image(ax2, reconstructed_images[i], "Output of CNN")

        plot_image(ax3, label_images[i], "Actual fracture image")

        plt.savefig(f"{save_path}/im{i+start_index}")
        plt.show()

    print("Images saved.")


def plot_image(ax, image, title):
    ax.imshow(tf.squeeze(image))
    ax.set_title(title)
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 


def train_model(x_train, y_train, model_name, loss_name):
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
    loss = sobel_loss if loss_name == "sobel" else ssim_loss

    artifact_remover.compile(loss=loss, optimizer=optim)
    artifact_remover.fit(x_train,
            y_train,
            epochs=200,
            shuffle=False,
            batch_size=10,
            verbose=2)
            #validation_data=(x_test, y_test))

    artifact_remover.save(f"./saved_model/{model_name}_{loss_name}_trained_model.h5")
    return artifact_remover

if __name__ == "__main__":

    if not os.path.exists("./images"):
        os.makedirs("./images/data")
        os.makedirs("./images/labels")
        os.makedirs("./images/pngs")

    if not os.path.exists("./saved_model/"):
        os.makedirs("./saved_model/")

    resize = False
    if (sys.argv[4] == "True"):
        resize = True

    model_name = sys.argv[1]
    loss_name = sys.argv[2]

    x_train, y_train, x_test, y_test = load_images("./images", 2500, 0.05, resize)
    
    if str(sys.argv[3]) == "load":
        artifact_remover = tf.keras.models.load_model(f"./saved_model/{model_name}_{loss_name}_trained_model.h5", compile=False)

        # set loss and optimizer here
        optim = keras.optimizers.Adam(learning_rate=0.001)
        loss = sobel_loss if loss_name == "sobel" else ssim_loss
        metrics = ["mse"]
        artifact_remover.compile(metrics=metrics, loss=loss, optimizer=optim)

    else:
        artifact_remover = train_model(x_train, y_train, model_name, loss_name)

    artifact_remover.evaluate(x_test, y_test)

    one_frac_x, one_frac_y = get_images("one_frac.npy", resize)
    point_scatter_x, point_scatter_y = get_images("point_scatter.npy", resize)
    ten_frac_x, ten_frac_y = get_images("ten_frac.npy", resize)
    two_close_x, two_close_y = get_images("two_close.npy", resize)

    x_special = np.stack([one_frac_x, point_scatter_x, ten_frac_x, two_close_x], axis=0)
    y_special = np.stack([one_frac_y, point_scatter_y, ten_frac_y, two_close_y], axis=0)

    emds = []
    mses = []
    psnrs = []
    ssims = []

    im_per_eval = 20

    for i in range(6):
        current_decoded_images = artifact_remover(x_test[i*im_per_eval:im_per_eval*i + im_per_eval+1])
        emds.append(calculate_emd(y_test[i*im_per_eval:im_per_eval*i + im_per_eval+1], current_decoded_images))
        mses.append(calculate_mse(y_test[i*im_per_eval:im_per_eval*i + im_per_eval+1], current_decoded_images))
        psnrs.append(calculate_psnr(y_test[i*im_per_eval:im_per_eval*i + im_per_eval+1], current_decoded_images))
        ssims.append(calculate_ssim(y_test[i*im_per_eval:im_per_eval*i + im_per_eval+1], current_decoded_images))

    special_images = artifact_remover(x_special)
    decoded_images = artifact_remover(x_test[:im_per_eval+1])

    print("Average earth mover distance: ", np.mean(emds))
    print("Average mean squared error: ", np.mean(mses))
    print("Average PSNR: ", np.mean(psnrs))
    print("Average SSIM: ", np.mean(ssims))

    plot_comparison(4, x_special, special_images, y_special, model_name, loss_name, 0)
    plot_comparison(im_per_eval, x_test[:im_per_eval+1], decoded_images, y_test[:im_per_eval+1], model_name, loss_name, 4)