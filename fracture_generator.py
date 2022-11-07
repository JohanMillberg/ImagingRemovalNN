import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import tensorflow as tf
from scipy.stats import truncnorm, norm, uniform
from scipy.signal import convolve2d as conv2
from PIL import Image

class FractureGenerator:

    def __init__(self,
                 image_width: int,
                 image_height: int,
                 n_fractures: int,
                 fracture_width,
                 buffer_size: int,
                 max_length: float = 160.0,
                 std_dev_length: float = 30.0,
                 std_dev_angle: float = 30,
                 max_iterations: int = 15):

        self.image_height = image_height
        self.image_width = image_width
        self.n_fractures = n_fractures
        self.fracture_width = fracture_width
        self.buffer_size = buffer_size
        self.max_length = max_length
        self.length_distribution = truncnorm(0, max_length, loc=max_length, scale=std_dev_length)
        self.angle_distribution = norm(loc=0, scale=std_dev_angle)
        self.max_iterations = max_iterations
        self.x_distribution = uniform(loc=0, scale=self.image_width)
        self.y_distribution = uniform(loc=0, scale=self.image_height)

    def generate_fractures(self, image: tf.Tensor):
        fracture_image = tf.zeros_like(image).numpy()
        for _ in range(self.n_fractures):
            fracture_is_valid = False
            n_iterations = 0

            while not fracture_is_valid and n_iterations < self.max_iterations: 
                fracture_length = self.length_distribution.rvs().astype(int)
                fracture_angle = self.angle_distribution.rvs()
                xs, ys = self._sample_coordinates()

                pixels_to_fracture = []

                while self._is_invalid_pixel(fracture_image, xs, ys):
                    xs, ys = self._sample_coordinates()                

                pixels_to_fracture.append((xs, ys))

                x_exact = xs
                y_exact = ys

                fractured_pixels = 1
                while fractured_pixels < fracture_length:
                    x_exact = x_exact + np.cos(fracture_angle)
                    y_exact = y_exact + np.sin(fracture_angle)

                    x_index = x_exact.astype(int)
                    y_index = y_exact.astype(int)

                    if self._is_invalid_pixel(fracture_image, x_index, y_index):
                        n_iterations += 1
                        break

                    if (x_index, y_index) not in pixels_to_fracture:
                        pixels_to_fracture.append((x_index, y_index))
                        fractured_pixels += 1

                fracture_is_valid = True
                self._create_buffer(fracture_image, pixels_to_fracture)
                for x, y in pixels_to_fracture:
                    self._fracture_pixel(fracture_image, x, y)

        fracture_image[fracture_image == -1] = 0
        fracture_image = self._blur_fracture_edges(fracture_image)
        fracture_image = tf.convert_to_tensor(fracture_image)
        resulting_image = tf.math.add(image, fracture_image)


        return resulting_image
    
    def _create_buffer(self, image, pixels_to_fracture):
        for x, y in pixels_to_fracture:
            for i in range(x - self.buffer_size, x + self.buffer_size):
                for j in range(y - self.buffer_size, y + self.buffer_size):
                    if not self._out_of_bounds(i, j):
                        image[i, j] = -1

    def _fracture_pixel(self, image, x, y):
        for i in range(x-int(self.fracture_width/2), x+int(self.fracture_width/2)):
            for j in range(y-int(self.fracture_width/2), y+int(self.fracture_width/2)):
                if not self._out_of_bounds(i, j) and not self._collides_with_fracture(image, i, j):
                    image[i, j] = 2
    
    def _blur_fracture_edges(self, image):
        convolved = image.copy()

        psf = np.array([[1, 2, 1], [2, 3, 2], [1, 2,1 ]], dtype='float32')
        psf *= 1 / np.sum(psf)

        convolved = conv2(convolved, psf, 'same')

        return convolved

    def _out_of_bounds(self, x, y):
        return x < 0 or x >= self.image_width or \
               y < 0 or y >= self.image_height

    def _sample_coordinates(self):
        xs = self.x_distribution.rvs().astype(int)
        ys = self.y_distribution.rvs().astype(int)
        return xs, ys

    def _is_invalid_pixel(self, image, x, y):
        if self._out_of_bounds(x, y):
            return True
        
        if self._collides_with_fracture(image, x, y):
            return True
        
        if self._pixel_in_buffer(image, x, y):
            return True

        return False

    def _collides_with_fracture(self, image, x, y):
        return image[x, y] == 2.0

    def _pixel_in_buffer(self, image, x, y):
        return image[x, y] == -1

    def plot_image(self, image):
        plt.gray()
        plt.imshow(tf.squeeze(image))
        plt.show()


def normalize_image(image: tf.Tensor):
    normalized = tf.cast(image, dtype=tf.float32)
    normalized = normalized / tf.reduce_max(tf.abs(normalized))

    return normalized


def main():
    image_height = 150
    image_width = 300
    n_fractures = 10
    fracture_width = 2
    buffer_size = 25
    max_length = 50
    std_dev_length = 10

    generator = FractureGenerator(image_height,
                                  image_width,
                                  n_fractures,
                                  fracture_width,
                                  buffer_size,
                                  max_length,
                                  std_dev_length)
    for i in range(5):
        result = generator.generate_fractures(tf.zeros(
                                            (image_height, image_width))
                                            ).numpy()

        result = normalize_image(result)
        result = tf.expand_dims(result, 2)
        # tf.keras.utils.save_img(
        #    f"./images/fractured/im{i}.jpg",
        #    result
        # )
    generator.plot_image(result)


if __name__ == "__main__":
    main()