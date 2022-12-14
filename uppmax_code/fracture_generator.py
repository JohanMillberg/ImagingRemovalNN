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
                 fractured_region_width: int,
                 fractured_region_height: int,
                 O_x: int,
                 O_y: int,
                 n_fractures: int,
                 fracture_width,
                 buffer_size: int,
                 max_length: float = 50.0,
                 min_length: float = 20.0,
                 std_dev_length: float = 10.0,
                 std_dev_angle: float = np.pi / 6.0,
                 mean_noise: float = 1.0,
                 std_dev_noise: float = 0.2,
                 max_iterations: int = 15,
                 background_velocity: float = 1.0
                 ):

        self.image_height = image_height
        self.image_width = image_width
        self.fractured_region_height = fractured_region_height
        self.fractured_region_width = fractured_region_width
        self.O_x = O_x
        self.O_y = O_y
        self.n_fractures = n_fractures
        self.fracture_width = fracture_width
        self.buffer_size = buffer_size
        self.max_length = max_length
        self.min_length = min_length
        self.background_velocity = background_velocity

        mean_length = (max_length + max_length) / 2
        a_length = (min_length - mean_length) / std_dev_length
        b_length = (max_length - mean_length) / std_dev_length

        self.length_distribution = truncnorm(a=a_length, b=b_length, loc=mean_length, scale=std_dev_length)
        #self.length_distribution = uniform(loc=min_length, scale=max_length)
        self.angle_distribution = norm(loc=-np.pi/2, scale=std_dev_angle)

        self.x_low = self.O_x
        self.x_high = self.x_low + self.fractured_region_width


        self.y_low = self.O_y
        self.y_high = self.y_low + self.fractured_region_height

        a_low = (0.3 - 0.45) / 0.05
        b_low = (0.6 - 0.45) / 0.05 
        self.low_velocity_modifier = truncnorm(a_low, b_low, loc=0.45, scale=0.05)

        a_high = (1.5 - 2.25) / 0.25
        b_high = (3.0 - 2.25) / 0.25
        self.high_velocity_modifier = truncnorm(a_high, b_high, loc=2.25, scale=0.25)
        self.modifier_distributions = [self.low_velocity_modifier, self.high_velocity_modifier]

        self.mean_noise = mean_noise
        self.std_dev_noise = std_dev_noise

        self.max_iterations = max_iterations

    def generate_fractures(self):
        fracture_image = np.full((self.image_height, self.image_width), self.background_velocity)
        for _ in range(self.n_fractures):
            n_iterations = 0
            fracture_is_valid = False
            selected_modifier = np.random.choice(self.modifier_distributions)
            modifier_value = selected_modifier.rvs()

            while (not fracture_is_valid) and (n_iterations < self.max_iterations): 
                fracture_length = self.length_distribution.rvs().astype(int)
                fracture_angle = self.angle_distribution.rvs()
                pixels_to_fracture = []

                # Sample a valid starting position for the fracture
                current_sample_iteration = 0
                xs, ys = self._sample_coordinates(current_sample_iteration)

                while self._is_invalid_pixel(fracture_image, xs, ys):
                    current_sample_iteration += 1
                    xs, ys = self._sample_coordinates(current_sample_iteration)                

                pixels_to_fracture.append((xs, ys))

                x_exact = xs
                y_exact = ys

                # Add the rest of the pixels to the fracture
                fractured_pixels = 1
                pixel_is_valid = True
                while fractured_pixels < fracture_length:
                    x_exact = x_exact + np.cos(fracture_angle)
                    y_exact = y_exact + np.sin(fracture_angle)

                    x_index = x_exact.astype(int)
                    y_index = y_exact.astype(int)

                    if self._is_invalid_pixel(fracture_image, x_index, y_index):
                        n_iterations += 1
                        pixel_is_valid = False
                        break

                    if (x_index, y_index) not in pixels_to_fracture:
                        pixels_to_fracture.append((x_index, y_index))
                        fractured_pixels += 1

                if not pixel_is_valid:
                    continue

                fracture_is_valid = True
                # Create the fracture
                self._create_buffer(fracture_image, pixels_to_fracture)
                for x, y in pixels_to_fracture:
                    self._fracture_pixel(fracture_image, x, y, modifier_value)

            if not fracture_is_valid:
                raise RuntimeError("Unable to fit fracture in image")

        # Produce the resulting image
        fracture_image[fracture_image == -1] = self.background_velocity # Remove the buffer
        # fracture_image = self._blur_fracture_edges(fracture_image)
        # fracture_image = self._add_noise(fracture_image, 1, 0.1)
        # fracture_image = tf.convert_to_tensor(fracture_image)
        # resulting_image = tf.math.add(image, fracture_image)
        fracture_image = fracture_image.reshape(self.image_width*self.image_height)

        return fracture_image
    
    def _create_buffer(self, image, pixels_to_fracture):
        for x, y in pixels_to_fracture:
            for i in range(x - self.buffer_size, x + self.buffer_size):
                for j in range(y - self.buffer_size, y + self.buffer_size):
                    if not self._out_of_bounds(i, j):
                        image[j, i] = -1

    def _fracture_pixel(self, image, x, y, modifier_value):
        for i in range(x-int(self.fracture_width/2), x+int(self.fracture_width/2)):
            for j in range(y-int(self.fracture_width/2), y+int(self.fracture_width/2)):
                image[j, i] = self.background_velocity*modifier_value
    
    def _blur_fracture_edges(self, image):
        convolved = image.copy()

        psf = np.array([[1, 2, 1], [2, 3, 2], [1, 2,1 ]], dtype='float32')
        psf *= 1 / np.sum(psf)

        convolved = conv2(convolved, psf, 'valid')

        return convolved

    def _out_of_bounds(self, x, y):
        return x < self.O_x or \
               x >= self.O_x + self.fractured_region_width or \
               y < self.O_y or \
               y >= self.O_y + self.fractured_region_height

    def _sample_coordinates(self, current_sample_iteration: int):
        if current_sample_iteration > self.max_iterations:
            raise RuntimeError("Unable to fit fracture in image")

        xs = int(np.random.uniform(self.x_low, self.x_high))
        ys = int(np.random.uniform(self.y_low, self.y_high))
        return xs, ys

    def _is_invalid_pixel(self, image, x, y):
        if self._out_of_bounds(x, y):
            return True
        
        elif self._collides_with_fracture(image, x, y):
            return True
        
        elif self._pixel_in_buffer(image, x, y):
            return True

        else:
            return False

    def _collides_with_fracture(self, image, x, y):
        collides = image[y, x] < self.background_velocity and image[y, x] > -1 \
            or image[y, x] > self.background_velocity

        return collides

    def _pixel_in_buffer(self, image, x, y):
        return image[y, x] == -1

    def _add_noise(self, image: np.array, mean_noise: int, std_dev_noise: int):
        gauss_noise = np.random.normal(loc=self.mean_noise,
                                       scale=self.std_dev_noise,
                                       size=image.size
                                       ).astype(np.float32)

        gauss_noise = gauss_noise.reshape(*image.shape)
        noisy_image = np.add(image, gauss_noise)

        return noisy_image
        """
        rvs_vec = np.vectorize(self._noise_helper,
                        excluded=['distribution'],
                        otypes=['float32'])
        return rvs_vec(p=image, distribution=self.noise_distribution)
        """

    def _noise_helper(self, p, distribution):
        return p + distribution.rvs()

    def plot_image(self, image_path):
        image = np.load(image_path)
        image = image.reshape(self.image_height, self.image_width, 1)
        plt.gray()
        # plt.imshow(tf.squeeze(image))
        plt.imshow(np.squeeze(image))
        plt.show()


def normalize_image(image: np.array):
    normalized = image.astype(np.float32)
    # normalized = normalized / tf.reduce_max(tf.abs(normalized))
    normalized = normalized / np.max(normalized)

    return normalized


def main():
    # Specify parameters
    image_height = 512
    image_width = 512
    fractured_region_height = 350
    fractured_region_width = 175
    O_x = 100
    O_y = 81
    n_fractures = 4
    fracture_width = 4
    buffer_size = 40 # space between fractures
    mean_noise = 1.0
    std_dev_noise = 0.2
    max_length = 50
    min_length = 20
    std_dev_length = 10
    std_dev_angle = np.pi / 6.0
    mean_noise = 1.0
    std_dev_noise = 0.2
    max_iterations = 15
    n_images_to_generate = 10
    background_velocity = 1000

    generator = FractureGenerator(image_width,
                                  image_height,
                                  fractured_region_width,
                                  fractured_region_height,
                                  O_x,
                                  O_y,
                                  n_fractures,
                                  fracture_width,
                                  buffer_size,
                                  max_length,
                                  min_length,
                                  std_dev_length,
                                  std_dev_angle,
                                  mean_noise,
                                  std_dev_noise,
                                  max_iterations,
                                  background_velocity)

    for i in range(n_images_to_generate):
        result = generator.generate_fractures()
    
if __name__ == "__main__":
    main()
