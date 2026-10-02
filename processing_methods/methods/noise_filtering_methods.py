import numpy as np

from methods.padding_methods import mirror


# -- Linear Filtering Methods --

# function takes an unpadded image, img, and a filter kernel, W
# performs convolution over the image with kernel W
def linear_filter(img, W):
    filter_size = W.shape[0]

    # the resulting filtered image should be a 2d array whose size is that of the noisy image
    # here I initialize it as a 2d array of zeros
    filtered_img = np.zeros((img.shape[0], img.shape[1])) 

    # make img a copy to avoid mutating the original image
    img = img.copy()
    # apply mirror padding to the noisy image
    img = mirror(img, k=(filter_size//2))


    for i in range(filtered_img.shape[0]):
        for j in range(filtered_img.shape[1]):
            # take a (filter_size x filter_size) window 
            window = img[i:filter_size+i, j:filter_size+j]
            
            # take the dot product of the flattened window and the flattened filter kernel
            # a (3x3) window is reshaped to (1, 9)
            # a (3x3) kernel is reshaped to (9, 1)
            # (1, 9) @ (9, 1) = (1) = an array with a single element
            # .item() makes it a scalar so it can be assigned to the filtered image
            filtered_img[i, j] = (window.reshape(1, -1) @ W.reshape(-1, 1)).item()

    # round the filtered image and clip it on [0, 255] then cast to type uint8
    return np.clip(np.round(filtered_img), 0, 255).astype(np.uint8)
 

# this function uses a smart filter kernel given in the function header by default but can also accept other smart filter kernels
# it simply takes a filter kernel and passes it to the linear_filter function
def smart_filter(img, W=(np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16)):
    return linear_filter(img, W)


# this function takes an image and a filter size and computes arithmetic mean filtering
# by passing a mean filter to the linear_filter function
def mean_filter(img, filter_size=3):
    W = (np.ones((filter_size, filter_size)) / filter_size**2)
    return linear_filter(img, W)



# -- Non-linear Filtering Methods --

# insertion sort is efficient for small number of elements.
# here I use it to arrange the variational series in an ascending order
# and since variational series of 3x3 window is small it is still efficient.
# I also kept track of the original indices so I can know which pixel is the pixel of interest (the center pixel)
def insertion_sort(A):
    indices = np.arange(A.shape[0])

    A = A.copy()
    i = 1
    while i < A.shape[0]:
        key = A[i]
        key_idx = i
        j = i - 1
        while j >= 0 and A[j] > key:
            A[j + 1] = A[j]
            indices[j + 1] = indices[j]
            j -= 1
        A[j + 1] = key
        indices[j + 1] = key_idx
        i += 1
        
    return A, indices


# auxillary function to take a window and flatten it into an unsorted variational series
def get_variational_series(window):
    # flatten the (n x m) window
    return window.reshape(-1)


# function takes as input an image and filter size and applies median filtering to the image
def median_filter(img, filter_size=3):
    # use a copy of the original image so as to not mutate the original image
    img = img.copy()

    # apply mirror padding to the image
    img = mirror(img, k=(filter_size//2))

    # a 2d array whose dimensions are the dimensions of the unpadded image
    filtered_img = np.zeros((img.shape[0] - filter_size + 1, img.shape[1] - filter_size + 1)) 

    # calculate the median index of the variational series
    median_index = int(np.ceil(filter_size / 2))

    # slide the filter over the noisy image from left to right, top to bottom.
    for i in range(filtered_img.shape[0]):
        for j in range(filtered_img.shape[1]):
            # find the variational series of the window
            variational_series = insertion_sort(get_variational_series(img[i:filter_size+i, j:filter_size+j]))
            # get the value that is the median of the variational series
            median = variational_series[median_index]

            # set the pixel at the center of the window to the median value
            filtered_img[i, j] = median

    # return the filtered image, rounded, clipped on [0, 255] and casted to type uint8 incase it was passed as a float
    return np.clip(np.round(filtered_img), 0, 255).astype(np.uint8)


def differential_rank_impulse_detection(img, filter_size=3, r=2, s=10):
    # make a copy of the image to avoid mutating the original image
    img = img.copy()

    # make another copy before padding
    # since this is a detection algorithm the undetected pixels should be left alone
    filtered_img = img.copy()

    # apply mirror padding to the noisy image
    img = mirror(img, (filter_size//2))

    # get the index of the median of the variational series
    median_index = filter_size**2 // 2

    # loop over the noisy image
    for i in range(img.shape[0] - filter_size + 1):
        for j in range(img.shape[1] - filter_size + 1):
            # take a (filter_size x filter_size) window 
            window = img[i:filter_size+i, j:filter_size+j]

            # get the variational series and the indices
            variational_series, indices = insertion_sort(get_variational_series(window))

            # to find the rank of the pixel centered in the window we can use the median_index which for (3x3) window is 5
            # for (5x5) window it would be 25//2 = 13
            # indices is like a record of how the original variational series elements moved
            # meaning that for a (3x3) window, value 5 in indices is the original center pixel before it was moved
            # we can find out where it is in the variational series by searching for its index:
            center_pixel_rank = -1
            for h in range(indices.shape[0]):
                if indices[h] == median_index:
                    center_pixel_rank = h
                    break

            # the median pixel rank of the sorted variational series is at the median
            median_pixel_rank = variational_series[median_index]
            
            # if the center pixel is on the left hand side of the variational series and within the bound r
            # we compare it to its neighbor on the right
            if center_pixel_rank >= 0 and center_pixel_rank < r:
                difference = np.abs(variational_series[center_pixel_rank] - variational_series[center_pixel_rank + 1])
            # likewise if the center pixel is on the right hand side and within the bound r, we compare it to its neighbor on the left
            elif center_pixel_rank >= variational_series.shape[0] - r and center_pixel_rank < variational_series.shape[0]:
                difference = np.abs(variational_series[center_pixel_rank] - variational_series[center_pixel_rank - 1])
            # in this case the center pixel is either at the median or not within the bound r so we don't touch it 
            else:
                continue
            
            # compare the difference to s, the threshold
            if difference >= s:
                # if the difference is at or above the threshold set the pixel to be the median of the variational series
                filtered_img[i, j] = variational_series[median_index]


    # return the filtered image, rounded and clipped on [0, 255] and casted to uint8
    return np.clip(np.round(filtered_img), 0, 255).astype(np.uint8)


# Notes I made while working on the implementation
# Differential Rank Impulse Detection
# r=3, s=10
#                          r                          r
#                     |---------|               |------------|
# frame.reshape(-1) = [25, 36, 36, 47, 201, 48, 104, 110, 120]
#                     [25, 36, 36, 47, 48, 104, 110, 120, 201 
#                                       M                  C
# Compare C = 201 to its neighbor on the left: 120
# 201 - 120 = 81
# 81 > s
# C = M = 48

