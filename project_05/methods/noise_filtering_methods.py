import numpy as np

from .padding_methods import pad
from .image_statistics import get_min_max


# Linear Filtering Methods
# function takes an unpadded image, img, and a filter kernel, W
# performs convolution over the image with kernel W
def linear_convolution_filter(img, W, mode="spatial"):
    # cast the image to float for operations
    img = img.astype(np.float64)
    window_size = W.shape[0]

    # the resulting filtered image should be a 2d array whose size is that of the noisy image
    # here I initialize it as a 2d array of zeros
    filtered_img = np.zeros((img.shape[0], img.shape[1])) 

    match mode:
        case "spatial":
            # apply mirror padding to the noisy image
            img = pad(img, pad_width=(window_size//2), mode="mirror")

            for i in range(filtered_img.shape[0]):
                for j in range(filtered_img.shape[1]):
                    # take a (window_size x window_size) window 
                    window = img[i:window_size+i, j:window_size+j]
                    
                    # take the dot product of the flattened window and the flattened filter kernel
                    # a (3x3) window is reshaped to (1, 9)
                    # a (3x3) kernel is reshaped to (9, 1)
                    # (1, 9) @ (9, 1) = (1) = an array with a single element
                    # .item() makes it a scalar so it can be assigned to the filtered image
                    filtered_img[i, j] = (window.reshape(1, -1) @ W.reshape(-1, 1)).item()
        case "frequency":
            h = np.zeros((img.shape[0], img.shape[1]))
            #h[:W.shape[0], :W.shape[1]] = W
            #H = np.fft.fft2(h)
            #G = np.fft.fft2(img)

            h[:2, :2] = W[1:, 1:]
            h[0:2, -1] = W[1:, 0]
            h[-1, :2] = W[0, 1:]
            h[-1, -1] = W[0, 0] 

            H = np.fft.fft2(h)
            G = np.fft.fft2(img)
                        
            filtered_img = (np.fft.ifft2(G * H)).real


    # round the filtered image and clip it on [0, 255] then cast to type uint8
    return np.clip(np.round(filtered_img), 0, 255).astype(np.uint8)
 

# this function uses a smart filter kernel given in the function header by default but can also accept other smart filter kernels
# it simply takes a filter kernel and passes it to the linear_filter function
def smart_filter(img, W=(np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16)):
    return linear_convolution_filter(img, W)


# this function takes an image and a filter size and computes arithmetic mean filtering
# by passing a mean filter to the linear_filter function
def mean_filter(img, window_size=3):
    
    return linear_convolution_filter(img, np.ones((window_size, window_size)) / window_size**2)



# Non-linear Filtering Methods
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
def median_filter(img, window_size=3):
    # use a copy of the original image so as to not mutate the original image
    img = img.copy()

    # apply mirror padding to the image
    img = pad(img, pad_width=(window_size//2), mode="mirror")

    # a 2d array whose dimensions are the dimensions of the unpadded image
    filtered_img = np.zeros((img.shape[0] - window_size + 1, img.shape[1] - window_size + 1)) 

    # calculate the median index of the variational series
    median_index = int(np.ceil(window_size / 2))

    # slide the filter over the noisy image from left to right, top to bottom.
    for i in range(filtered_img.shape[0]):
        for j in range(filtered_img.shape[1]):
            # find the variational series of the window
            variational_series = insertion_sort(get_variational_series(img[i:window_size+i, j:window_size+j]))
            # get the value that is the median of the variational series
            median = variational_series[median_index]

            # set the pixel at the center of the window to the median value
            filtered_img[i, j] = median

    # return the filtered image, rounded, clipped on [0, 255] and casted to type uint8 incase it was passed as a float
    return np.clip(np.round(filtered_img), 0, 255).astype(np.uint8)


def drid(img, r=2, s=10, filter_method="median", window_size=3, kernel=None, ):
    # make a copy of the image to avoid mutating the original image
    img = img.astype(np.float64)

    # make another copy before padding
    # since this is a detection algorithm the undetected pixels should be left alone
    filtered_img = img.copy()

    # apply mirror padding to the noisy image
    img = pad(img, pad_width=(window_size//2), mode="mirror")

    # get the index of the median of the variational series
    median_index = window_size**2 // 2

    # loop over the noisy image
    for i in range(filtered_img.shape[0]):
        for j in range(filtered_img.shape[1]):
            # take a (window_size x window_size) window 
            window = img[i:window_size+i, j:window_size+j]

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
                match filter_method:
                    # if the difference is at or above the threshold apply filtering to the pixel of interest
                    case "median":
                        filtered_img[i, j] = variational_series[median_index]
                    case "linear_convolution":
                        filtered_img[i, j] = (window.reshape(1, -1) @ W.reshape(-1, 1)).item()
                    case "mean":
                        W = np.ones((window_size, window_size)) * window_size**-2
                        filtered_img[i, j] = (window.reshape(1, -1) @ W).item()
                    case "smart_mean":
                        filtered_img[i, j] = (window.reshape(1, -1) @ W.reshape(-1, 1)).item()

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


def get_boolean_layers(img):
    # get the maximum intensity in the image
    _, max_intensity = get_min_max(img)

    # cast to int to avoid indexing issues
    max_intensity = int(max_intensity)

    # there are 0 to max_intensity, binary planes
    binary_planes = np.arange(max_intensity + 1)
    
    # create the stacks as binary planes of the same size as the image,
    # where pixels in a binary plane are 1 if the corresponding pixel in the original image
    # is greater than or equal to the index of that binary plane and 0 otherwise.
    # can do this quickly and efficiently using numpy broadcasting
    stack = img[:, :, None] >= binary_planes
    # set the 0th binary plane to all 0s
    stack[:, :, 0] = False

    # return the stack of binary planes
    return stack


def threshold_boolean_filter(img, window_size=3, t=5):
    window_radius = window_size // 2
    n_neighbors = window_size**2 - 1
    H, W = img.shape


    stack = get_boolean_layers(pad(img, pad_width=window_radius, mode="mirror")).astype(np.int32)
    
    window_sum = sum(
            stack[di:di + H, dj:dj + W, :]
            for di in range(window_size)
            for dj in range(window_size)
    )

    center = stack[window_radius:window_radius + H, window_radius:window_radius+W, :]
    ones = window_sum - center
    disagree = np.where(center == 1, n_neighbors - ones, ones)
    filtered = center.astype(bool) ^ (disagree >= t)   # XOR: flip where disagree >= t

    return np.clip(np.round(filtered.sum(axis=-1)), 0, 255).astype(np.uint8)
