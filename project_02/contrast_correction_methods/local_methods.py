import numpy as np
from PIL import Image

from mirroring.mirror import mirror
from image_statistics.stats import *


def get_local_cumulative_histogram(frame):
    n = 0
    center_pixel = frame[frame.shape[0] // 2, frame.shape[1] // 2]

    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i, j] <= center_pixel:
                n += 1
    return n


def local_histogram_equalization(img, filter_size=3, new_min=20, new_max=235):
    img = img.copy()

    # k determines how much padding is necessary and is related to the size of the filter
    # filter_size (// is integer division) 2
    k = filter_size // 2
    # add mirror padding to the image
    padded_img = mirror(img, k=k)
   
    d = (new_max - new_min) / filter_size**2

    # iterate over the rows
    for i in range(img.shape[0]):
        # iterate over the columns
        for j in range(img.shape[1]):
            # cut a frame from the padded image
            frame = padded_img[i:filter_size+i, j:filter_size+j]
            # get the pixel at the center of the frame
            center_pixel = frame[frame.shape[0] // 2, frame.shape[1] // 2]

            # apply local histogram equalization:
            # np.count_nonzero(frame <= center_pixel) is the number of pixels less than or equal to the center pixel
            img[i, j] = new_min + np.count_nonzero(frame <= center_pixel) * d
            
            # my version of the above. it uses my function get_local_cumulative_histogram()
            # which returns a count of the pixels less than or equal to the center pixel.
            # it is a bit slower than numpy's np.count_nonzero() but gives the same results
            #img[i, j] = new_min + get_local_cumulative_histogram(frame) * d:

    # return the image with rounded pixels, clipped to the range [0, 255] and casted to uint8
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def local_linear_contrast_correction(img, filter_size=3, new_mean=127, new_std=60):
    img = img.copy()
    k = filter_size // 2
    padded_img = mirror(img, k=k)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            frame = padded_img[i:filter_size+i, j:filter_size+j]
            #local_mean = get_mean(frame)
            #local_std = get_std(frame, variance=get_variance(img, mean=local_mean))

            # had to use numpy's built in mean and std function since the
            # mean and std are calculated for every frame and it was very slow.
            # numpy uses C/C++ loops to get the mean and std so its much
            # faster compared to python loops since C/C++ is a compiled language
            local_mean = frame.mean()
            local_std = frame.std()
            
            if local_std == 0:
                img[i, j] = new_mean
            else:
                a = new_std / local_std
                b = new_mean - local_mean * a
            
                img[i, j] = (a * img[i, j]) + b

    return np.clip(np.round(img), 0, 255).astype(np.uint8)
