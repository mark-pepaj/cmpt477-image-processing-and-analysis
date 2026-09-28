import numpy as np
from PIL import Image

from noise_filtering_methods.tests import test_linear_filter, test_nonlinear_filter

# read in an image as a numpy array and as type float64
#img = np.asarray(Image.open("Lena_Y.tif")).astype(np.float64)
img = np.asarray(Image.open("Airplane-F16_Y.tif")).astype(np.float64)


# mean filter
#W = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

# smart filter
W = np.array([[1, 3, 1], [3, 7, 3], [1, 3, 1]]) / 23


# this function generates the specified noise model and applies the linear filter given by W.
# then it measures MSE, RMSE, and PSNR of the clean image with the noisy image and the clean image with the filtered image.
# if show=True is given, then the function displays the noisy image and the filtered image.
test_linear_filter(img, W, noise_model="Gaussian", std_fraction=0.2, show=False)
#test_nonlinear_filter(img, noise_model="Salt and Pepper", p=0.005, show=False)
