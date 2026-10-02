import numpy as np

from methods.padding_methods import mirror
from methods.noise_filtering_methods import insertion_sort

# this function calculates the mean over a window
def get_window_mean(window):
    sum = 0
    # flatten the window
    window = window.reshape(-1)

    # loop over the flattened window indices and sum each element up
    for i in range(window.shape[0]):
        sum += window[i]

    # divide by the size of the window
    return sum / (window.shape[0])


def get_window_median(window):
    variational_series, _ = insertion_sort(window.reshape(-1))
    return variational_series[int(np.ceil(window.shape[0] / 2))]


# this function applies unshark masking and takes as arguments the image, the coefficient, k, and the size of the local window
def unsharp_mask_sharpening(img, filter_size=3, k=1.0, averaging="arithmetic mean"):
    sharpened_img = img.astype(np.float64)
    img = img.astype(np.float64)
    img = mirror(img, filter_size//2)
    
    for i in range(sharpened_img.shape[0]):
        for j in range(sharpened_img.shape[1]):
            if averaging == "median":
                g = img[i, j] - get_window_median(img[i:filter_size+i, j:filter_size+j])
            else:
                g = img[i, j] - get_window_mean(img[i:filter_size+i, j:filter_size+j])

            sharpened_img[i, j] = img[i, j] + k * g
   
    return np.clip(np.round(sharpened_img), 0, 255).astype(np.uint8)


def global_frequency_correction(img, filter_width=3, filter_height=3, k1=0.5, k2=1, k3=0.5, averaging="arithmetic mean"):
    enhanced_img = img.astype(np.float64)
    img = img.astype(np.float64)
    img = mirror(img, filter_width//2)

    for i in range(enhanced_img.shape[0]):
        for j in range(enhanced_img.shape[1]):
            window = img[i:filter_width+i, j:filter_height+j]
            if averaging == "median":
                g = (k1 * img[i, j]) + (k2 * (img[i, j] - get_window_mean(window))) + (k3 * get_window_mean(window))
            else:
                g = (k1 * img[i, j]) + (k2 * (img[i, j] - get_window_median(window))) + (k3 * get_window_median(window))
            
            enhanced_img[i, j] = g

    return np.clip(np.round(enhanced_img), 0, 255).astype(np.uint8)
