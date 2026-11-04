import numpy as np
from PIL import Image

from methods.image_statistics import *
from methods.noise_generation_methods import *
from methods.noise_filtering_methods import *
from methods.sharpening_methods import *


#img = np.asarray(Image.open("DATA/images/Lake_Y.tif"))
img = np.asarray(Image.open("DATA/images/Temple_Y.tif"))

#sharpened_img = unsharp_mask(img=img, window_size=3, k=1.5, mode="arithmetic mean")
#Image.fromarray(sharpened_img).show()
#enhanced_img = global_frequency_correction(img=img, window_width=3, window_height=3, k1=0.5, k2=3, k3=0.5, mode="arithmetic mean")
#Image.fromarray(enhanced_img).show()

#enhanced_img = laplacian_1(img, direction="downward")
#enhanced_img = laplacian_1(img, direction="upward")
#enhanced_img = laplacian_2(img, direction="downward")
#enhanced_img = laplacian_2(img, direction="upward")
#Image.fromarray(enhanced_img).show()

#noisy_img = salt_pepper_impulse_noise(img, p=0.01)
#filtered_img = drid(img)
#Image.fromarray(noisy_img).show()
#Image.fromarray(filtered_img).show()

