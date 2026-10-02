from pathlib import Path
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

from methods.image_statistics import *


# -- Global Contrast Correction Methods -- 
def histogram_equalization(img, new_min=20, new_max=235, max_val=256):
    histogram = get_histogram(img, show=False, save_path=None) 

    S = np.zeros(256, dtype=float)  

    d = (new_max - new_min) / (img.shape[0] * img.shape[1])

    S[0], S[-1] = new_min + histogram[0] * d, new_max
    
    for i in range(1, max_val):
        for j in range(i+1):
            S[i] += histogram[j]
        S[i] = S[i] * d + new_min

    S = np.clip(np.round(S), 0, max_val-1).astype(np.uint8)
    return S[img.astype(np.uint8)]


def linear_contrast_correction(img, new_mean, new_std, old_mean=None, old_std=None, max_val=256):
    if old_mean is None and old_std is None:
        old_mean = get_mean(img)
        old_std = get_std(img, variance=get_variance(img, mean=old_mean))
    elif old_mean is None and old_std is not None:
        old_mean = get_mean(img)
    elif old_mean is not None and old_std is None:
        old_std = get_std(img, variance=get_variance(img, mean=old_mean))
    

    S = np.zeros(256, dtype=float)

    a = new_std / old_std
    b = new_mean - old_mean * a
    
    for i in range(256):
        S[i] = (a * i) + b

    S = np.clip(np.round(S), 0, max_val-1).astype(np.uint8) 

    return S[img.astype(np.uint8)]


def plot_histograms(original_img, corrected_img, show=True, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    original_img_histogram = get_histogram(original_img, show=False, save_path=None)
    corrected_img_histogram = get_histogram(corrected_img, show=False, save_path=None)
    
    ax1.bar(np.arange(256), original_img_histogram)
    ax1.set_title("Original Image Histogram")

    ax2.bar(np.arange(256), corrected_img_histogram)
    ax2.set_title("Image Histogram after Correction")

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()
    plt.close(fig)


def correct_with_histogram_equalization(img, new_min, new_max, save_path=None, silent=False):
    if not silent:
        print("Image Stats After Histogram Equalization:")
    img_he = histogram_equalization(img, new_min=new_min, new_max=new_max)
    img_he_stats = get_stats(img_he, silent=silent)
    
    if save_path is not None:
        plot_histograms(img, img_he, show=silent, save_path=save_path + "original_vs_he_histograms")
        Image.fromarray(img_he).save(save_path + "he_img.png")
    else:
        plot_histograms(img, img_he, show=silent, save_path=save_path)

    return img_he_stats


def correct_with_linear_contrast_correction(img, new_mean, new_std, save_path=None, silent=False):
    if not silent:
        print("Image Stats After Linear Contrast Correction:")
    img_llc = linear_contrast_correction(img, new_mean=new_mean, new_std=new_std)
    img_llc_stats = get_stats(img_llc, silent=silent)

    if save_path is not None:
        plot_histograms(img, img_llc, show=silent, save_path=save_path + "original_vs_llc_histograms")
        Image.fromarray(img_llc).save(save_path + "llc_img.png")
    else:
        plot_histograms(img, img_llc, show=silent, save_path=save_path)

    return img_llc_stats


def local_contrast_correction(img, filter_size=3, correction_mode=None, new_min=0, new_max=255, new_mean=127, new_std=60, save_path=None, silent=False, max_val=256):
    corrected_img = np.zeros((img.shape[0] - filter_size + 1, img.shape[1] - filter_size + 1))
    for i in range(img.shape[0] - filter_size + 1):
        for j in range(img.shape[1] - filter_size + 1):
            if correction_mode == "he":
                corrected_img[i:filter_size + i, j:filter_size + j] = histogram_equalization(img[i:filter_size + i, j:filter_size + j], new_min=new_min, new_max=new_max)
            elif correction_mode == "llc":
                corrected_img[i:filter_size + i, j:filter_size + j] = linear_contrast_correction(img[i:filter_size + i, j:filter_size + j], new_mean=new_mean, new_std=new_std)
    return np.clip(np.round(corrected_img), 0, max_val-1).astype(np.uint8)


def image_analysis_and_correction(img_path=None, correction_mode="he, llc", new_min=0, new_max=255, new_mean=127, new_std=60, save_path=None, silent=False):
    if img_path is None:
        print("Path to image file expected but not received")
        return
    
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)

    img = np.asarray(Image.open(img_path)).astype(np.float64)

    if not silent:
        print("Original Image Stats:")
    img_stats = get_stats(img, silent=silent)

    if correction_mode == "he":
        return (img_stats,
                correct_with_histogram_equalization(img=img, new_min=new_min, new_max=new_max, save_path=save_path, silent=silent))
    elif correction_mode == "llc": 
        return (img_stats,
                correct_with_linear_contrast_correction(img=img, new_mean=new_mean, new_std=new_std, save_path=save_path, silent=silent))
    elif correction_mode == "he, llc":
        return (img_stats, correct_with_histogram_equalization(img=img, new_min=new_min, new_max=new_max, save_path=save_path, silent=silent),
                correct_with_linear_contrast_correction(img=img, new_mean=new_mean, new_std=new_std, save_path=save_path, silent=silent))
    else:
        raise ValueError(f"Unknown correction_mode: {correction_mode!r} (expected 'he', 'llc', or 'he, llc')")


# -- Local Contrast Correction Methods --
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
