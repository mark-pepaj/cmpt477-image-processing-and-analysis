import numpy as np
from PIL import Image

def mirror(img, k=1):
    img = img.copy()

    left_edge = np.fliplr(img[:, 1:k+1])
    right_edge = np.fliplr(img[:, (-1 - k):-1])
    img = np.append(left_edge, img, axis=1)
    img = np.append(img, right_edge, axis=1)

    top_edge = np.flipud(img[1:k+1, :])
    bottom_edge = np.flipud(img[(-1 - k):-1, :])
    img = np.append(top_edge, img, axis=0)
    img = np.append(img, bottom_edge, axis=0)

    return img


