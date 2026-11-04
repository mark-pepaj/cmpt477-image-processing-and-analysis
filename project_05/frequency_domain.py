import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt

from methods.image_statistics import *
from methods.padding_methods import pad

def h_window(height, width, var):
    h = np.zeros((height, width))
    for i in range(height): 
        for j in range(width): 
            window = blurred_window[i:window_size+i, j:window_size+j]
            h[i, j] = math.exp(-(i**2 + j**2) / var) / (2 * math.pi * var)
    return h

def main():
    offset = 32 
    window_size = 256
    img = np.asarray(Image.open("DATA/images/Lena_Y.tif")).astype(np.float64)#[offset : window_size + offset, offset : window_size + offset]
    

    fft = np.fft.fft2(img)
    #shifted_fft = np.fft.fftshift(fft)
    #magnitude = np.abs(shifted_fft)
    #log_magnitude = np.log1p(magnitude)


    var = get_variance(img)
    window_size = 3
    h = img.copy()
    #blurred_img = pad(img, pad_width=window_size//2, mode="mirror")

    for i in range(img.shape[0]): 
        for j in range(img.shape[1]): 
            #window = blurred_window[i:window_size+i, j:window_size+j]
            h[i, j] = math.exp(-(i**2 + j**2) / var) / (2 * math.pi * var)

    blurred_img = np.fft.ifft2(fft @ h.reshape(-1)).real

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(blurred_img, cmap="gray")
    axes[1].set_title("Magnitude Spectrum")
    axes[1].axis("off")
    plt.tight_layout()

    plt.show() 
    plt.close()

if __name__ == "__main__":
    main()
