from pathlib import Path
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

def get_min_max(img):
    min, max = float('inf'), float('-inf')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < min:
                min = img[i, j]
            if img[i, j] > max:
                max = img[i, j]

    return min, max


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
    if variance is not None:
        return variance**0.5
    var = get_variance(img)
    return var**0.5


def get_SNR(img, mean=None, std=None):
    if mean is None and std is not None:
        mean = get_mean(img)
    elif mean is not None and std is None:
        std = get_std(img, mean=mean)
    elif mean is None and std is None:
        mean = get_mean(img)
        std = get_std(img, mean=mean)
     
    return mean / std


def get_stats(img, silent):
    min, max = get_min_max(img)
    mean = get_mean(img)
    variance = get_variance(img, mean=mean)
    standard_deviation = get_std(img, variance=variance)
    SNR = get_SNR(img, mean=mean, std=standard_deviation)

    if not silent:
        print(f"image shape ({img.shape[0]}, {img.shape[1]})")
        print(f"min: {min}  max: {max}")
        print(f"mean: {mean}")
        print(f"variance: {variance}")
        print(f"standard deviation: {standard_deviation}")
        print(f"SNR: {SNR}")
        print("\n")

    return min, max, mean, variance, standard_deviation, SNR


def get_histogram(img, show=True, save_path=None):
    histogram = np.zeros(256, dtype=int)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            histogram[int(img[i, j])] += 1
    
    fig = plt.figure()
    plt.bar(np.arange(256), histogram)
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    
    return histogram


def histogram_equalization(img, new_min, new_max):
    histogram = get_histogram(img, show=False, save_path=None) 

    S = np.zeros(256, dtype=float)  

    S[0], S[-1] = new_min, new_max

    d = (new_max - new_min) / (img.shape[0] * img.shape[1])
    
    for i in range(1, 256):
        for j in range(i+1):
            S[i] += histogram[j]
        S[i] = S[i] * d + S[0]

    S = np.clip(S, 0, 255).astype(np.uint8)

    return S[img.astype(np.uint8)]

def linear_contrast_correction(img, new_mean, new_std):
    old_mean, old_std = get_mean(img), get_std(img)
    
    S = np.zeros(256, dtype=float)
    
    a = new_std / old_std
    b = new_mean - old_mean * a
    
    for i in range(256):
        S[i] = (a * i) + b

    S = np.clip(S, 0, 255).astype(np.uint8) 

    return S[img.astype(np.uint8)]


def plot_histograms(original_img, corrected_img, show=True, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    original_img_histogram = get_histogram(original_img, show=False, save_path=None)
    corrected_img_histogram = get_histogram(corrected_img, show=False, save_path=None)
    
    ax1.bar(np.arange(256), original_img_histogram)
    ax1.set_title("Original Image Histogram")

    ax2.bar(np.arange(256), corrected_img_histogram)
    ax2.set_title("Image Histogram after Correction")

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


def image_analysis_and_correction(img_path, save_path, silent=False):
    if img_path is None:
        print("Path to image file expected but not received")
        return
    
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)

    img = np.asarray(Image.open(img_path)).astype(np.float64)

    print("Original Image Stats:")
    img_stats = [get_stats(img, silent=silent)]

    print("Image Stats After Histogram Equalization:")
    img_he = histogram_equalization(img, new_min=20, new_max=235)
    img_he_stats = [get_stats(img_he, silent=silent)]
    

    print("Image Stats After Linear Contrast Correction:")
    img_llc = linear_contrast_correction(img, new_mean=127, new_std=60)
    img_llc_stats = [get_stats(img_llc, silent=silent)] 

    if save_path is not None:
        plot_histograms(img, img_he, show=silent, save_path=save_path + "original_vs_he_histograms")
        plot_histograms(img, img_llc, show=silent, save_path=save_path + "original_vs_llc_histograms")
        
        img_he = Image.fromarray(img_he)
        img_llc = Image.fromarray(img_llc)

        img_he.save(save_path + "he_img.png")
        img_llc.save(save_path + "llc_img.png")
    else:
        plot_histograms(img, img_he, show=silent, save_path=save_path)
        plot_histograms(img, img_llc, show=silent, save_path=save_path)


def main():
    image_analysis_and_correction("LowContrast-a.tif", "results/LowContrast-a/")


if __name__ == "__main__":
    main()
