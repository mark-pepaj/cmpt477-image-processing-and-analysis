import numpy as np

from padding_methods.pad import mirror

# function takes an unpadded image, img, and a filter kernel, W
# performs convolution over the image with kernel W
def linear_filter(img, W):
    filter_size = W.shape[0]

    # the resulting filtered image should be a 2d array whose size is that of the noisy image
    # here I initialize it as a 2d array of zeros
    filtered_img = np.zeros((img.shape[0], img.shape[1])) 

    # make img a copy to avoid mutating the original image
    img = img.astype(np.float64)
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
    #return np.clip(np.round(filtered_img), 0, 255).astype(np.uint8)
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

