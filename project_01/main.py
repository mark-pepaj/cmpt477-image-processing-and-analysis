from pathlib import Path
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

# Problem 1
# min, max, mean, standard deviation, variance, SNR
# I designed several functions to carry out each calculation and then created get_stats(img) which calls each function and returns their outputs
def get_min_max(img):
    # initialize min as very large and very small numbers, respectively
    min, max = float('inf'), float('-inf')

    # img.shape returns the shape of the 2-d array in the form: [m, n]
    # iterate down the rows
    for i in range(img.shape[0]):
        # iterate through each column in the row
        for j in range(img.shape[1]):
            # if the pixel's value is smaller than min, assign it as the new min
            if img[i, j] < min:
                min = img[i, j]
            # if the pixel's value is larger than the max, assign it as the new max
            if img[i, j] > max:
                max = img[i, j]

    # return min and max
    return min, max


def get_mean(img):
    mean = 0.0
    # iterate across the rows
    for i in range(img.shape[0]):
        # iterate across each column of the row
        for j in range(img.shape[1]):
            # sum up each pixel's value
            mean += img[i, j]
    # average the sum by the size of the image
    return mean / (img.shape[0] * img.shape[1])


def get_variance(img):
    mean = get_mean(img)
    var = 0 
    # iterate across the rows
    for i in range(img.shape[0]):
        # iterate across each column of the row
        for j in range(img.shape[1]):
            # sum up the square of each pixel's distance from the mean
            var += (img[i, j] - mean)**2
    # average the sum by the size of the image
    return var / (img.shape[0] * img.shape[1])


def get_std(img):
    # standard deviation is found by taking the square root of the variance
    var = get_variance(img)
    return var**0.5


def get_SNR(img):
    # SNR is approximated as mean / standard_deviation
    mean, standard_deviation = get_mean(img), get_std(img)
    return mean / standard_deviation


# function to call intermediate functions and return their outputs
# if silent is true, the function will not print an output
def get_stats(img, silent):
    # here each function is called on the image
    min, max = get_min_max(img)
    mean = get_mean(img)
    variance = get_variance(img)
    standard_deviation = get_std(img)
    SNR = get_SNR(img)

    # if silent is False then print the statistics
    if not silent:
        print(f"image shape ({img.shape[0]}, {img.shape[1]})")
        print(f"min: {min}  max: {max}")
        print(f"mean: {mean}")
        print(f"variance: {variance}")
        print(f"standard deviation: {standard_deviation}")
        print(f"SNR: {SNR}")
        print("\n")

    # also return the min, max, mean, etc.
    return min, max, mean, variance, standard_deviation, SNR


# Problem 2
# Histogram Visualisation
# if show=True, the historam will be displayed at runtime
# if a save path is given, the histogram will be saved as an image to the path
def get_histogram(img, show=True, save_path=None):
    # initialize an vector of zeros with length 256 where each element denotes the frequency that an intensity occurs
    histogram = np.zeros(256, dtype=int)

    # iterate through each row
    for i in range(img.shape[0]):
        # iterate through each column in a row
        for j in range(img.shape[1]):
            # use the pixel value as the index for the histogram and add 1 for each intensity to count the occurence
            histogram[int(img[i, j])] += 1
    
    # create the figure using matplotlib.pyplot library
    fig = plt.figure()
    plt.bar(np.arange(256), histogram)
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    # save the figure if a save path is given
    if save_path is not None:
        plt.savefig(save_path)
    # if show == True then display the figure
    if show:
        plt.show()
    plt.close(fig)
    
    # also return the histogram vector
    return histogram


# Problem 3
def histogram_equalization(img, new_min, new_max):
    # get the image's histogram
    histogram = get_histogram(img, show=False, save_path=None) 

    # initialize a vector of zeros whose length is the number of intensities
    # this will be used to create a mapping from original intensities to the equalized histogram intensities
    S = np.zeros(256, dtype=float)  

    # S_0 and S_K are given as arguments to the function
    S[0], S[-1] = new_min, new_max

    # I factored out this computation since it is appears in the loop
    d = (new_max - new_min) / (img.shape[0] * img.shape[1])
    
    # for each intensity in the range S_0+1 to S_K-1, calculate S_i
    for i in range(1, 256):
        # S_i is the sum of the number of times the jth intensity occurs in the image for j=0 to i
        for j in range(i+1):
            S[i] += histogram[j]
        # the sum is scaled by d = (S_0 - S_K) / (MN) and S[0] is added to it
        S[i] = S[i] * d + S[0]

    # restrict the mapping to [0, 255] and cast to uint8 since the original image is 8-bit
    S = np.clip(S, 0, 255).astype(np.uint8)

    # return the image, mapped to each new intensity
    return S[img.astype(np.uint8)]

# Problem 4
# Linear Contrast Correction
# function accepts a desired mean and desired standard deviation
def linear_contrast_correction(img, new_mean, new_std):
    # calculate the current mean and standard deviation of the image
    old_mean, old_std = get_mean(img), get_std(img)
    
    # initialize a vector of as many elements as there are intensities in the image 
    # will be used to create a mapping from old intensities to the corrected intensities
    S = np.zeros(256, dtype=float)
    
    # factor out this computation to avoid recomputing in the loop
    a = new_std / old_std
    b = new_mean - old_mean * a
    
    # iterate over each intensity and build the mapping from r_i to S_i
    for i in range(256):
        S[i] = (a * i) + b

    # restrict the mapping to [0, 255] and cast to uint8 since the original image is 8-bit
    S = np.clip(S, 0, 255).astype(np.uint8) 

    # return the image, mapped to each new intensity
    return S[img.astype(np.uint8)]


# function to plot the original image's histogram and the corrected image's histogram
def plot_histograms(original_img, corrected_img, show=True, save_path=None):
    # create a figure containing two subplots, vertically stacked
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # compute the histogram of the original image and the corrected image
    original_img_histogram = get_histogram(original_img, show=False, save_path=None)
    corrected_img_histogram = get_histogram(corrected_img, show=False, save_path=None)
    
    # plot both histograms with their corresponding titles
    ax1.bar(np.arange(256), original_img_histogram)
    ax1.set_title("Original Image Histogram")

    ax2.bar(np.arange(256), corrected_img_histogram)
    ax2.set_title("Image Histogram after Correction")

    # save the figure if a path is given
    if save_path is not None:
        plt.savefig(save_path)

    # if show == True then display the figure
    if show:
        #plt.tight_layout()
        plt.show()
    plt.close(fig)


# this function takes as arguments the path to the image file, a save path, and a boolean variable, silent
# if img_path is None then the function will display a corresponding message and exit
# if save_path is None, the corrected images and histograms will not be saved
# if silent is false, the outputs will be silenced and nothing will be printed
def image_analysis_and_correction(img_path, save_path, silent=False):
    if img_path is None:
        print("Path to image file expected but not received")
        return
    
    # if save_path contains a directory then it will be created if it doesn't exist
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)

    # read the image as a Numpy array
    # I cast it to float64 so later when performing operations on it the intermediate variables are inferred to be float64 and not uint8
    img = np.asarray(Image.open(img_path)).astype(np.float64)

    print("Original Image Stats:")
    # a python list of the original image's stats
    img_stats = [get_stats(img, silent=silent)]

    print("Image Stats After Histogram Equalization:")
    img_he = histogram_equalization(img, new_min=30, new_max=235)
    # a python list of the stats after the equalization
    img_he_stats = [get_stats(img_he, silent=silent)]
    

    print("Image Stats After Linear Contrast Correction:")
    img_llc = linear_contrast_correction(img, new_mean=127, new_std=60)
    # a python list of the stats after the correction
    img_llc_stats = [get_stats(img_llc, silent=silent)] 

    # if the save_path is given then save the histograms and the new images
    if save_path is not None:
        plot_histograms(img, img_he, show=silent, save_path=save_path + "original_vs_he_histograms")
        plot_histograms(img, img_llc, show=silent, save_path=save_path + "original_vs_llc_histograms")
        
        # convert the corrected image arrays to Image objects
        img_he = Image.fromarray(img_he)
        img_llc = Image.fromarray(img_llc)

        # save the images
        img_he.save(save_path + "he_img.png")
        img_llc.save(save_path + "llc_img.png")
    else:
    # if not then just plot the histograms
        plot_histograms(img, img_he, show=silent, save_path=save_path)
        plot_histograms(img, img_llc, show=silent, save_path=save_path)


def main():
    # LowContrast-a
    image_analysis_and_correction("../../DATA/images/LowContrast-a.tif", "results/LowContrast-a/")

    # LowContrast-b
    image_analysis_and_correction("../../DATA/images/LowContrast-b.tif", "results/LowContrast-b/")


if __name__ == "__main__":
    main()
