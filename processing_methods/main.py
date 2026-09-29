import numpy as np
from PIL import Image

from sharpening_methods.methods import unsharp_mask_sharpening
from image_statistics.stats import *

#img = np.asarray(Image.open("Lake_Y.tif"))
img = np.asarray(Image.open("Temple_Y.tif"))

sharpened_img = unsharp_mask_sharpening(img, filter_size=3, k=1.1, averaging="arithmetic mean")
compare_images(img, sharpened_img)

sharpened_img = unsharp_mask_sharpening(img, filter_size=3, k=1.5, averaging="arithmetic mean")
compare_images(img, sharpened_img)

