from pathlib import Path
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

def get_min_max(img):
    minimum, maximum = float('inf'), float('-inf')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < minimum:
                minimum = img[i, j]
            if img[i, j] > maximum:
                maximum = img[i, j]

    return minimum, maximum


def get_mean(img):
    mean = 0.0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            mean += img[i, j]
    return mean / (img.shape[0] * img.shape[1])


def get_variance(img, mean=None):
    if mean is None:
        mean = get_mean(img)
    var = 0 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            var += (img[i, j] - mean)**2
    return var / (img.shape[0] * img.shape[1])


def get_std(img, variance=None):
    # standard deviation is found by taking the square root of the variance
    if variance is None:
        variance = get_variance(img)
    return variance**0.5


def get_SNR(img, mean=None, standard_deviation=None):
    # SNR is approximated as mean / standard_deviation
    if mean is None and standard_deviation is None:
        mean = get_mean(img)
        standard_deviation = get_std(img, variance=get_variance(img, mean=mean))
    elif mean is None and standard_deviation is not None:
        mean = get_mean(img)
    elif mean is not None and standard_deviation is None:
        standard_deviation = get_std(img, variance=get_variance(img, mean=mean))
    return mean / standard_deviation


def get_MSE(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    MSE = 0
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            MSE += (img1[i, j] - img2[i, j])**2
    return MSE / (img1.shape[0] * img1.shape[1])


def get_RMSE(img1, img2, MSE=None):
    if MSE is None:
        MSE = get_MSE(img1, img2)
    return MSE**0.5


def get_PSNR(img1, img2, RMSE=None, max_val=255):
    if RMSE is None:
        RMSE = get_RMSE(img1, img2)
    if RMSE == 0:
        return float('inf')
    return 20 * np.log10((max_val) / RMSE)


def compare_images(img1, img2, silent=False):
    MSE = get_MSE(img1, img2)
    RMSE = get_RMSE(img1, img2, MSE=MSE)
    PSNR = get_PSNR(img1, img2, RMSE=RMSE)
    
    if not silent:
        print(f"MSE: {MSE}")
        print(f"RMSE: {RMSE}")
        print(f"PSNR: {PSNR}")

    return MSE, RMSE, PSNR
    

def get_stats(img, silent=False):
    minimum, maximum = get_min_max(img)
    mean = get_mean(img)
    variance = get_variance(img, mean=mean)
    standard_deviation = get_std(img, variance=variance)
    SNR = get_SNR(img, mean=mean, standard_deviation=standard_deviation)

    if not silent:
        print(f"image shape ({img.shape[0]}, {img.shape[1]})")
        print(f"min: {minimum}  max: {maximum}")
        print(f"mean: {mean}")
        print(f"variance: {variance}")
        print(f"standard deviation: {standard_deviation}")
        print(f"SNR: {SNR}")
        print("\n")

    return minimum, maximum, mean, variance, standard_deviation, SNR

def get_histogram(img, show=True, save_path=None):
    histogram = np.zeros(256, dtype=int)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            histogram[int(img[i, j])] += 1
    return histogram


    """ 
    #code for plotting the histogram

    fig = plt.figure()
    plt.bar(np.arange(256), histogram)
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    """ 

