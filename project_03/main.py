import numpy as np
from PIL import Image

from image_statistics.stats import *
from padding_methods.pad import mirror
from noise_generation.generate import *
from noise_filtering_methods.nonlinear_filters import *
from noise_filtering_methods.tests import test_nonlinear_filter


clean_img = np.asarray(Image.open("images/Lena_Y.tif"))

"""
Image.fromarray(impulse_noise(clean_img, noise_model="random", p=0.05, n_min=0, n_max=255)).save("results/corrupted_images/Lena_random_noise_005.png")
Image.fromarray(impulse_noise(clean_img, noise_model="random", p=0.1, n_min=0, n_max=255)).save("results/corrupted_images/Lena_random_noise_010.png")
Image.fromarray(impulse_noise(clean_img, noise_model="salt and pepper", p=0.05)).save("results/corrupted_images/Lena_sp_noise_005.png")
Image.fromarray(impulse_noise(clean_img, noise_model="salt and pepper", p=0.1)).save("results/corrupted_images/Lena_sp_noise_010.png")

random_noisy_img_005 = np.asarray(Image.open("results/corrupted_images/Lena_random_noise_005.png"))
random_noisy_img_010 = np.asarray(Image.open("results/corrupted_images/Lena_random_noise_010.png"))
sp_noisy_img_005 = np.asarray(Image.open("results/corrupted_images/Lena_sp_noise_005.png"))
sp_noisy_img_010 = np.asarray(Image.open("results/corrupted_images/Lena_sp_noise_010.png"))

Image.fromarray(differential_rank_impulse_detection(random_noisy_img_005, filter_size=3, r=2, s=10)).save("results/filtered_images/random_filtered_img_005.png")
Image.fromarray(differential_rank_impulse_detection(random_noisy_img_010, filter_size=3, r=2, s=10)).save("results/filtered_images/random_filtered_img_010.png")
Image.fromarray(differential_rank_impulse_detection(sp_noisy_img_005, filter_size=3, r=2, s=10)).save("results/filtered_images/sp_filtered_img_005.png")
Image.fromarray(differential_rank_impulse_detection(sp_noisy_img_010, filter_size=3, r=2, s=10)).save("results/filtered_images/sp_filtered_img_010.png")

random_filtered_img_005 = np.asarray(Image.open("results/filtered_images/random_filtered_img_005.png"))
random_filtered_img_010 = np.asarray(Image.open("results/filtered_images/random_filtered_img_010.png"))
sp_filtered_img_005 = np.asarray(Image.open("results/filtered_images/sp_filtered_img_005.png"))
sp_filtered_img_010 = np.asarray(Image.open("results/filtered_images/sp_filtered_img_010.png"))

_, RMSE_noisy_clean, PSNR_noisy_clean = compare_images(clean_img, random_noisy_img_005, silent=True)
_, RMSE_filtered_clean, PSNR_filtered_clean = compare_images(clean_img, random_filtered_img_005, silent=True)
print("random noise (p = 0.05) DRID filtering")
print(f"noisy vs clean    | RMSE: {RMSE_noisy_clean:.4f} - PSNR: {PSNR_noisy_clean:.4f}")
print(f"filtered vs clean | RMSE: {RMSE_filtered_clean:.4f} - PSNR: {PSNR_filtered_clean:.4f}\n")

_, RMSE_noisy_clean, PSNR_noisy_clean = compare_images(clean_img, random_noisy_img_010, silent=True)
_, RMSE_filtered_clean, PSNR_filtered_clean = compare_images(clean_img, random_filtered_img_010, silent=True)
print("random noise (p = 0.1) DRID filtering")
print(f"noisy vs clean    | RMSE: {RMSE_noisy_clean:.4f} - PSNR: {PSNR_noisy_clean:.4f}")
print(f"filtered vs clean | RMSE: {RMSE_filtered_clean:.4f} - PSNR: {PSNR_filtered_clean:.4f}\n")

_, RMSE_noisy_clean, PSNR_noisy_clean = compare_images(clean_img, sp_noisy_img_005, silent=True)
_, RMSE_filtered_clean, PSNR_filtered_clean = compare_images(clean_img, sp_filtered_img_005, silent=True)
print("salt and pepper noise (p = 0.05) DRID filtering")
print(f"noisy vs clean    | RMSE: {RMSE_noisy_clean:.4f} - PSNR: {PSNR_noisy_clean:.4f}")
print(f"filtered vs clean | RMSE: {RMSE_filtered_clean:.4f} - PSNR: {PSNR_filtered_clean:.4f}\n")

_, RMSE_noisy_clean, PSNR_noisy_clean = compare_images(clean_img, sp_noisy_img_010, silent=True)
_, RMSE_filtered_clean, PSNR_filtered_clean = compare_images(clean_img, sp_filtered_img_010, silent=True)
print("salt and pepper noise (p = 0.1) DRID filtering")
print(f"noisy vs clean    | RMSE: {RMSE_noisy_clean:.4f} - PSNR: {PSNR_noisy_clean:.4f}")
print(f"filtered vs clean | RMSE: {RMSE_filtered_clean:.4f} - PSNR: {PSNR_filtered_clean:.4f}\n")
"""


noisy_img = np.asarray(Image.open("results/filtered_images/sp_filtered_img_010.png"))
Image.fromarray(differential_rank_impulse_detection(noisy_img, filter_size=3, r=2, s=40)).save("results/filtered_images/sp_filtered_img_010.png")
sp_filtered_img_010 = np.asarray(Image.open("results/filtered_images/sp_filtered_img_010.png"))
_, RMSE_noisy_clean, PSNR_noisy_clean = compare_images(clean_img, noisy_img, silent=True)
_, RMSE_filtered_clean, PSNR_filtered_clean = compare_images(clean_img, sp_filtered_img_010, silent=True)

print("salt and pepper noise (p = 0.10) DRID filtering")
print(f"noisy vs clean    | RMSE: {RMSE_noisy_clean:.4f} - PSNR: {PSNR_noisy_clean:.4f}")
print(f"filtered vs clean | RMSE: {RMSE_filtered_clean:.4f} - PSNR: {PSNR_filtered_clean:.4f}\n")

