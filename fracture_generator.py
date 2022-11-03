import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import truncnorm, norm, uniform

class FractureGenerator:

    def __init__(self,
                 image_width: int,
                 image_height: int,
                 n_fractures: int,
                 max_length: float = 160.0,
                 std_dev_length: float = 30.0,
                 std_dev_angle: float = 30,
                 max_iterations: int = 10):

        self.image_height = image_height
        self.image_width = image_width
        self.n_fractures = n_fractures
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
                for x, y in pixels_to_fracture:
                    fracture_image[x, y] = 2
            
        fracture_image = tf.convert_to_tensor(fracture_image)
        resulting_image = tf.math.add(image, fracture_image)
        return resulting_image

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

        return False

    def _collides_with_fracture(self, image, x, y):
        return image[x, y] > 0

    def plot_image(self, image):
        plt.gray()
        plt.imshow(tf.squeeze(image))
        plt.show()

generator = FractureGenerator(150, 300, 10, 50, 10)
result = generator.generate_fractures(tf.zeros((150, 300))).numpy()
result = result / 3
generator.plot_image(result)