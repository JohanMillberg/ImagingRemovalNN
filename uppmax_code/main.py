from forwardsolver_gpu import ForwardSolver
from fracture_generator import FractureGenerator
import numpy as np
import cupy as cp
import sys, getopt
from cupyx.scipy import ndimage
from os.path import exists
import time

def generate_I_matrix(solver, image_index, N_x_im, N_y_im, N_t, N_s, plot, save):
	start = time.time()
	c = cp.load(f"/proj/i_rom/images/labels/im{image_index}.npy")
	I = solver.calculate_intensity(c)
	
	if exists(f"./I_0_{N_t}_{N_s}.npy"):
		I_0 = cp.load(f"./I_0_{N_t}_{N_s}.npy")
	else:
		R = cp.load(f"./R_0_{N_t}_{N_s}.npy")	
		I_0 = solver.calculate_imaging_func(R)

	I = I - I_0

	I = get_image_derivative(N_y_im, N_x_im, I)

	if plot:
		solver.plot_intensity(I, f"im{image_index}")

	if save:
		cp.save(f"/proj/i_rom/images/data/im{image_index}.npy", I )

	finish = time.time()

	print(f"I created in {finish - start} seconds")

def get_image_derivative(N_y_im, N_x_im, I):
	I = I.reshape((N_y_im, N_x_im))
	dx = cp.array([[1, 0, -1]])
	I_x = ndimage.convolve(I, dx)
	I_x = I_x.reshape((-1, 1))

	return I_x
	
def main(argv):
	n_images_to_generate = int(argv[1])
	print(f"Generating {n_images_to_generate} images.")

	plot = False

	save = True
	
	image_height = 512
	image_width = 512
	fractured_region_height = 350
	fractured_region_width = 175
	O_x = 30
	O_y = 81
	n_fractures = 4
	fracture_width = 4
	buffer_size = 35 # space between fractures
	mean_noise = 1.0
	std_dev_noise = 0.2
	max_length = 50
	min_length = 20
	std_dev_length = 10
	std_dev_angle = 30.0
	mean_noise = 1.0
	std_dev_noise = 0.2
	max_iterations = 15
	background_velocity = 1000
	N_t = 90
	N_s = 50

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

	print("Generator instantiated...")

	solver = ForwardSolver( 
				N_x = image_width,
				N_y = image_height,
				N_s = N_s,
				delta_x = 0.0063,
				tau = 3.0303*10**(-5),
				N_t = N_t,
				background_velocity_value = background_velocity,
				Bsrc_file = "Bsrc_T.txt",
				N_x_im = fractured_region_width,
				N_y_im = fractured_region_height,
				O_x = O_x,
				O_y = O_y
	)

	print("Solver instantiated...")

	generated_images = 0
	while generated_images < n_images_to_generate:
		print(f"Image number {generated_images}") 
		result = generator.generate_fractures()
		np.save(f"/proj/i_rom/images/labels/im{generated_images}.npy", result)

		try:
			generate_I_matrix(solver, generated_images, fractured_region_width, fractured_region_height, N_t, N_s, plot, save)
			generated_images += 1
		except np.linalg.LinAlgError:
			print("Mass matrix was singular, recalculating...")	


if __name__ == "__main__":
	main(sys.argv)
