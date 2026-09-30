import numpy as np
from PIL import Image

from noise_filtering_methods.tests import test_linear_filter

# read in an image as a numpy array and as type float64
#img = np.asarray(Image.open("images/Lena_Y.tif"))
img = np.asarray(Image.open("images/Airplane-F16_Y.tif")).astype(np.float64)


# mean filter
W = np.ones((3, 3), dtype=np.float64) / 9


# smart filter
#W = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16
#W = np.array([[1, 3, 1], [3, 7, 3], [1, 3, 1]], dtype=np.float64) / 23


# this function generates the specified noise model and applies the linear filter given by W.
# then it measures MSE, RMSE, and PSNR of the clean image with the noisy image and the clean image with the filtered image.
# if show=True is given, then the function displays the noisy image and the filtered image.

test_linear_filter(img, W, noise_model="Gaussian", std_fraction=0.3, show=False)

#test_linear_filter(img, W, noise_model="Gaussian", std_fraction=0.2, show=True)
#test_linear_filter(img, W, noise_model="Gaussian", std_fraction=0.2, show=True)
