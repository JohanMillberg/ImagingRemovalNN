import tensorflow as tf
import numpy as np
from cnn import calculate_emd, calculate_mse, calculate_ssim, calculate_sobel_metric, load_images, sobel_loss, ssim_loss, get_images, plot_comparison
import sys, getopt


if __name__=="__main__":
    model_name = sys.argv[1]
    loss_name = sys.argv[2]
    stride = int(sys.argv[3])

    resize = False
    if (stride == 2):
        resize = True

    x_train, y_train, x_test, y_test = load_images("./images", 2500, 0.2, resize)

    artifact_remover = tf.keras.models.load_model(f"./saved_model/{model_name}_{loss_name}_{stride}_trained_model.h5", compile=False)
    # set loss and optimizer here
    optim = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = sobel_loss if loss_name == "sobel" else ssim_loss
    metrics = ["mse"]
    artifact_remover.compile(metrics=metrics, loss=loss, optimizer=optim)

    one_frac_x, one_frac_y = get_images("one_frac.npy", resize)
    point_scatter_x, point_scatter_y = get_images("point_scatter.npy", resize)
    ten_frac_x, ten_frac_y = get_images("ten_frac.npy", resize)
    two_close_x, two_close_y = get_images("two_close.npy", resize)

    x_special = np.stack([one_frac_x, point_scatter_x, ten_frac_x, two_close_x], axis=0)
    y_special = np.stack([one_frac_y, point_scatter_y, ten_frac_y, two_close_y], axis=0)

    images_used_to_evaluate = 500
    
    special_images = artifact_remover(x_special)
    output = artifact_remover(x_test[:images_used_to_evaluate])

    emd = calculate_emd(y_test[:images_used_to_evaluate], output)
    mse = calculate_mse(y_test[:images_used_to_evaluate], output)
    ssim = calculate_ssim(y_test[:images_used_to_evaluate], output)
    sobel = calculate_sobel_metric(y_test[:images_used_to_evaluate], output)

    print("Average earth mover distance: ", emd)
    print("Average mean squared error: ", mse)
    print("Average SSIM: ", ssim)
    print("Average sobel loss: ", sobel)

    plot_comparison(4, x_special, special_images, y_special, model_name, loss_name, stride, 0)
    plot_comparison(20, x_test[:21], output[:21], y_test[:21], model_name, loss_name, stride, 4)