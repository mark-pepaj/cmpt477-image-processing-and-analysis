import numpy as np
from PIL import Image

from methods.image_statistics import *
from methods.padding_methods import mirror
from methods.noise_generation_methods import *
from methods.noise_filtering_methods import *


clean_img = np.asarray(Image.open("images/Lena_Y.tif"))

# corrupt the images
random_noisy_img_05p = impulse_noise(clean_img, noise_model="random", p=0.05, n_min=0, n_max=255)
random_noisy_img_10p = impulse_noise(clean_img, noise_model="random", p=0.1, n_min=0, n_max=255)
sp_noisy_img_05p = impulse_noise(clean_img, noise_model="salt and pepper", p=0.05)
sp_noisy_img_10p = impulse_noise(clean_img, noise_model="salt and pepper", p=0.1)

# save the corrupted images
Image.fromarray(random_noisy_img_05p).save("results/corrupted_images/Lena_random_noise_05p.png")
Image.fromarray(random_noisy_img_10p).save("results/corrupted_images/Lena_random_noise_10p.png")
Image.fromarray(sp_noisy_img_05p).save("results/corrupted_images/Lena_sp_noise_05p.png")
Image.fromarray(sp_noisy_img_10p).save("results/corrupted_images/Lena_sp_noise_10p.png")


# filter the images
random_filtered_img_05p = differential_rank_impulse_detection(random_noisy_img_05p, filter_size=3, r=2, s=10)
random_filtered_img_10p = differential_rank_impulse_detection(random_noisy_img_10p, filter_size=3, r=2, s=10)
sp_filtered_img_05p = differential_rank_impulse_detection(sp_noisy_img_05p, filter_size=3, r=2, s=10)
sp_filtered_img_10p = differential_rank_impulse_detection(sp_noisy_img_10p, filter_size=3, r=2, s=10)

# save the filtered images
Image.fromarray(random_filtered_img_05p).save("results/filtered_images/random_filtered_img_05p.png")
Image.fromarray(random_filtered_img_10p).save("results/filtered_images/random_filtered_img_10p.png")
Image.fromarray(sp_filtered_img_05p).save("results/filtered_images/sp_filtered_img_05p.png")
Image.fromarray(sp_filtered_img_10p).save("results/filtered_images/sp_filtered_img_10p.png")


# print the RMSE and PSNR for the corrupt and filtered images
_, RMSE_noisy_clean, PSNR_noisy_clean = compare_images(clean_img, random_noisy_img_05p, silent=True)
_, RMSE_filtered_clean, PSNR_filtered_clean = compare_images(clean_img, random_filtered_img_05p, silent=True)
print("random noise (p = 0.05) DRID filtering")
print(f"noisy vs clean    | RMSE: {RMSE_noisy_clean:.4f} - PSNR: {PSNR_noisy_clean:.4f}")
print(f"filtered vs clean | RMSE: {RMSE_filtered_clean:.4f} - PSNR: {PSNR_filtered_clean:.4f}\n")

_, RMSE_noisy_clean, PSNR_noisy_clean = compare_images(clean_img, random_noisy_img_10p, silent=True)
_, RMSE_filtered_clean, PSNR_filtered_clean = compare_images(clean_img, random_filtered_img_10p, silent=True)
print("random noise (p = 0.1) DRID filtering")
print(f"noisy vs clean    | RMSE: {RMSE_noisy_clean:.4f} - PSNR: {PSNR_noisy_clean:.4f}")
print(f"filtered vs clean | RMSE: {RMSE_filtered_clean:.4f} - PSNR: {PSNR_filtered_clean:.4f}\n")

_, RMSE_noisy_clean, PSNR_noisy_clean = compare_images(clean_img, sp_noisy_img_05p, silent=True)
_, RMSE_filtered_clean, PSNR_filtered_clean = compare_images(clean_img, sp_filtered_img_05p, silent=True)
print("salt and pepper noise (p = 0.05) DRID filtering")
print(f"noisy vs clean    | RMSE: {RMSE_noisy_clean:.4f} - PSNR: {PSNR_noisy_clean:.4f}")
print(f"filtered vs clean | RMSE: {RMSE_filtered_clean:.4f} - PSNR: {PSNR_filtered_clean:.4f}\n")

_, RMSE_noisy_clean, PSNR_noisy_clean = compare_images(clean_img, sp_noisy_img_10p, silent=True)
_, RMSE_filtered_clean, PSNR_filtered_clean = compare_images(clean_img, sp_filtered_img_10p, silent=True)
print("salt and pepper noise (p = 0.1) DRID filtering")
print(f"noisy vs clean    | RMSE: {RMSE_noisy_clean:.4f} - PSNR: {PSNR_noisy_clean:.4f}")
print(f"filtered vs clean | RMSE: {RMSE_filtered_clean:.4f} - PSNR: {PSNR_filtered_clean:.4f}\n")
