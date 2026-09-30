import numpy as np

from image_statistics.stats import *
from padding_methods.pad import mirror
from noise_generation.generate import *
from noise_filtering_methods.linear_filters import *
from noise_filtering_methods.nonlinear_filters import *

# both functions work in the same way but use different filtering methods

# this function accepts an image, a filter kernel, a specification for a noise model
# optional arguments are: an std dev fraction for guassian noise generation
# a percentage of effected pixels for impulse noise, bounds for random noise, and a min and max value for bipolar noise
# show can be set to True to display the noisy and filtered images
def test_linear_filter(img, W, noise_model="gaussian", std_fraction=0.1, p=0.01, n1=0, n2=255, n_min=0, n_max=255, show=False):
    filter_size = W.shape[0]

    # generate the noisy image based on the noise_model
    if noise_model == "gaussian":
        # calculate the mean and std of the image for Gaussian noise
        _, _, mean, _, std, _ = get_stats(img, silent=True)
        noisy_img = gaussian_noise(img, mean=mean, std=std_fraction * std)
    elif noise_model == "salt and pepper":
        noisy_img = salt_pepper_impulse_noise(img, p=p)
    elif noise_model == "bipolar":
        noisy_img = bipolar_impulse_noise(img, p=p, n1=n1, n2=n2)
    elif noise_model == "random":
        noisy_img = random_impulse_noise(img, p=p, n_min=n_min, n_max=n_max)
    
    # apply linear filtering with kernel, W
    filtered_img = linear_filter(noisy_img, W)

    # use compare_images, a function I designed to calculate the MSE, RMSE, and PSNR of the images
    _, RMSE_noisy_clean, PSNR_noisy_clean = compare_images(noisy_img, img, silent=True)
    _, RMSE_filtered_clean, PSNR_filtered_clean = compare_images(filtered_img, img, silent=True)
   
    # if show is True then display the clean, noisy, and filtered images
    if show:
        Image.fromarray(img).show()
        Image.fromarray(noisy_img).show()
        Image.fromarray(filtered_img).show()

    # print the RMSE and PSNR values of the images
    print(f"noisy vs clean    | RMSE: {RMSE_noisy_clean:.4f} - PSNR: {PSNR_noisy_clean:.4f}")
    print(f"filtered vs clean | RMSE: {RMSE_filtered_clean:.4f} - PSNR: {PSNR_filtered_clean:.4f}")
    
    # return the filtered image
    return filtered_img


# same for this function but it applies non linear filtering and accepts a filter method as an argument
def test_nonlinear_filter(img, noise_model="salt and pepper", filter_method="Differential Rank", filter_size=3, r=2, s=10, std_fraction=0.1, p=0.005, n1=0, n2=255, n_min=0, n_max=255, show=False):
    if noise_model == "gaussian":
        _, _, mean, _, std, _ = get_stats(img, silent=True)
        noisy_img = gaussian_noise(img, mean=mean, std=std_fraction * std)
    elif noise_model == "salt and pepper":
        noisy_img = salt_pepper_impulse_noise(img, p=p)
    elif noise_model == "bipolar":
        noisy_img = bipolar_impulse_noise(img, p=p, n1=n1, n2=n2)
    elif noise_model == "random":
        noisy_img = random_impulse_noise(img, p=p, n_min=n_min, n_max=n_max)
    
    if filter_method == "median":
        filtered_img = median_filter(noisy_img, filter_size=filter_size)
    else:
        filtered_img = differential_rank_impulse_detection(noisy_img, r=r, s=s)

    # caculate the RMSE and PSNR of the images
    _, RMSE_noisy_clean, PSNR_noisy_clean = compare_images(noisy_img, img, silent=True)
    _, RMSE_filtered_clean, PSNR_filtered_clean = compare_images(filtered_img, img, silent=True)
   
    # if show is True, display the images
    if show:
        Image.fromarray(img).show()
        Image.fromarray(noisy_img).show()
        Image.fromarray(filtered_img).show()

    # print the RMSE and PSNR of the images
    print(f"noisy vs clean    | RMSE: {RMSE_noisy_clean:.4f} - PSNR: {PSNR_noisy_clean:.4f}")
    print(f"filtered vs clean | RMSE: {RMSE_filtered_clean:.4f} - PSNR: {PSNR_filtered_clean:.4f}")

    
    # return the filtered image
    return filtered_img



