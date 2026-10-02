import numpy as np
from PIL import Image

from methods.sharpening_methods import *
from methods.edge_detection_methods import *
from methods.image_statistics import *

#img = np.asarray(Image.open("DATA/images/Lake_Y.tif"))
img = np.asarray(Image.open("DATA/images/Temple_Y.tif"))

#sharpened_img = unsharp_mask_sharpening(img, filter_size=3, k=1.5, averaging="arithmetic mean")
#Image.fromarray(sharpened_img).show()

#enhanced_img = global_frequency_correction(img=img, filter_width=3, filter_height=3, k1=0.5, k2=3, k3=0.5, averaging="arithmetic mean")
#Image.fromarray(enhanced_img).show()


#kernel1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#kernel2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
#print(kernel1.reshape(1, -1).shape)
#print(kernel2.shape)

#enhanced_img = laplacian_1(img, direction="upward")

#enhanced_img = laplacian_2(img, direction="downward")
enhanced_img = laplacian_2(img, direction="upward")
Image.fromarray(enhanced_img).show()
