import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from methods.image_statistics import *
from methods.padding_methods import mirror
from methods.noise_generation_methods import *
from methods.noise_filtering_methods import *


# rank order ev with mean
def mean_rank_order_ev_filter(img, filter_size=3, ev=25):
    # cast the image array to float64
    img = img.astype(np.float64)
    filtered_img = img.copy() 

    # apply mirror padding to the image array
    img = mirror(img, k=(filter_size//2))

    # loop over rows of the image array
    for i in range(filtered_img.shape[0]):
        # loop over columns of the image array
        for j in range(filtered_img.shape[1]):
            # the ev interval starts empty for each window
            ev_interval = []

            # take a window
            window = img[i : filter_size + i, j : filter_size + j]

            # get the pixel of interest
            center_pixel = window.reshape(-1)[(filter_size**2)//2]

            # get the variational series
            variational_series = get_variational_series(window)
            
            # loop over the variational series
            for k in range(len(variational_series)):
                # if the absolute value of the difference between the pixel of interest and
                # the elements of the variational series is less than or equal to the threshold ev then add it to the ev interval
                if np.abs(center_pixel - variational_series[k]) <= ev:
                    ev_interval.append(variational_series[k])
            # finally update the current pixel to be the mean of the ev interval
            filtered_img[i, j] = get_mean(ev_interval)

    # round and clip the image on [0, 255] and cast to uin8
    return np.clip(np.round(filtered_img), 0, 255).astype(np.uint8)


# rank order ev with median
def median_rank_order_ev_filter(img, er=4):
    # filter_size is fixed at 3x3 for simplicity
    filter_size = 3

    # er can be 1, 2, 3, or 4
    if er <= 0 or er > 4:
        er = 4 

    # cast the image array to float64
    img = img.astype(np.float64)
    filtered_img = img.copy() 
    
    # apply mirror padding to the noisy image
    img = mirror(img, k=(filter_size//2))

    # loop over rows of noisy image
    for i in range(filtered_img.shape[0]):
        # loop over columns of noisy image
        for j in range(filtered_img.shape[1]):
            # er interval starts empty for each window
            er_interval = []

            # get the window
            window = img[i : filter_size + i, j : filter_size + j]

            # get the sorted variational series and the original indices after sorting
            variational_series, indices = insertion_sort(get_variational_series(window))

            # the rank of the pixel of interest is the index of indices where indices == (filter_size**2 / 2)
            center_pixel_rank = np.where(indices == filter_size**2//2)[0][0]

            # loop over the variational series
            for k in range(len(variational_series)):
                # if absolute value of R(pixel of interest) - the rank (k) of the elements in the variational series
                # is within the threshold then append k to the er interval
                if np.abs(center_pixel_rank - k) <= er:
                    er_interval.append(variational_series[k])
            # get the median of the er interval
            filtered_img[i, j] = er_interval[len(er_interval) // 2]
    
    # return the rounded filtered image clipped on [0, 255] and casted to uin8
    return np.clip(np.round(filtered_img), 0, 255).astype(np.uint8)


# Rank Order Ev filter with Differential Rank Impulse Detection
# same as rank_order_ev_filter but uses DRID to detect noisy pixels then uses rank_order_er_filtering
def DRID_median_rank_order_er_filter(img, r=2, s=10, er=4):
    if er <= 0 or er > 4:
        er = 4 
    filter_size = 3

    img = img.astype(np.float64)
    filtered_img = img.copy() 
    img = mirror(img, k=(filter_size//2))

    # get the index of the median of the variational series
    median_index = filter_size**2 // 2

    # loop over the noisy image
    for i in range(img.shape[0] - filter_size + 1):
        for j in range(img.shape[1] - filter_size + 1):
            # take a (filter_size x filter_size) window 
            window = img[i:filter_size+i, j:filter_size+j]

            # get the variational series and the indices
            variational_series, indices = insertion_sort(get_variational_series(window))

            # to find the rank of the pixel centered in the window we can use the median_index which for (3x3) window is 5
            # for (5x5) window it would be 25//2 = 13
            # indices is like a record of how the original variational series elements moved
            # meaning that for a (3x3) window, value 5 in indices is the original center pixel before it was moved
            # we can find out where it is in the variational series by searching for its index:
            center_pixel_rank = -1
            for h in range(indices.shape[0]):
                if indices[h] == median_index:
                    center_pixel_rank = h
                    break

            # if the center pixel is on the left hand side of the variational series and within the bound r
            # we compare it to its neighbor on the right
            if center_pixel_rank >= 0 and center_pixel_rank < r:
                difference = np.abs(variational_series[center_pixel_rank] - variational_series[center_pixel_rank + 1])
            # likewise if the center pixel is on the right hand side and within the bound r, we compare it to its neighbor on the left
            elif center_pixel_rank >= variational_series.shape[0] - r and center_pixel_rank < variational_series.shape[0]:
                difference = np.abs(variational_series[center_pixel_rank] - variational_series[center_pixel_rank - 1])
            # in this case the center pixel is either at the median or not within the bound r so we don't touch it 
            else:
                continue
            
            # compare the difference to s, the threshold
            if difference >= s:
                # if the difference is at or above the threshold set the pixel to be the median of the er interval
                er_interval = []
                for k in range(len(variational_series)):
                    if np.abs(center_pixel_rank - k) <= er:
                        er_interval.append(variational_series[k])
                filtered_img[i, j] = er_interval[len(er_interval) // 2]


    # return the filtered image, rounded and clipped on [0, 255] and casted to uint8
    return np.clip(np.round(filtered_img), 0, 255).astype(np.uint8)





def main():
    # the source image
    img = np.asarray(Image.open("images/Lena_Y.tif"))

    # Rank Order EV Filter against Gaussian Noise with sigma=0.2 and sigma=0.3
    """
    mean = get_mean(img)
    std = get_std(img)
    
    noisy_img_g02 = gaussian_noise(img, mean=mean, std=0.2 * std)
    noisy_img_g03 = gaussian_noise(img, mean=mean, std=0.3 * std)

    filtered_img_g02 = mean_rank_order_ev_filter(img=noisy_img_g02, filter_size=3, ev=get_std(noisy_img_g02))
    filtered_img_g03 = mean_rank_order_ev_filter(img=noisy_img_g03, filter_size=3, ev=get_std(noisy_img_g03))

    _, noisy_RMSE_g02, noisy_PSNR_g02 = compare_images(img, noisy_img_g02, silent=True)
    _, noisy_RMSE_g03, noisy_PSNR_g03 = compare_images(img, noisy_img_g03, silent=True)

    _, filtered_RMSE_g02, filtered_PSNR_g02 = compare_images(img, filtered_img_g02, silent=True)
    _, filtered_RMSE_g03, filtered_PSNR_g03 = compare_images(img, filtered_img_g03, silent=True)

    # Plotting of the noisy and filtered images for Gaussian noise with sigma=0.2
    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Rank Order EV Filter\n(Ev=noisy image's std_dev)")
    axs[0].imshow(noisy_img_g02, cmap="gray", vmin=0, vmax=255)
    axs[0].set_title(f"Guassian (s=0.2)\nRMSE={noisy_RMSE_g02:.3f}\nPSNR={noisy_PSNR_g02:.3f}")

    axs[1].imshow(filtered_img_g02, cmap="gray", vmin=0, vmax=255)
    axs[1].set_title(f"Filtered Image\nRMSE={filtered_RMSE_g02:.3f}\nPSNR={filtered_PSNR_g02:.3f}")

    for ax in axs:
        ax.set_axis_off()

    plt.show()


    # Plotting of the noisy and filtered images for Gaussian noise with sigma=0.3
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(noisy_img_g03, cmap="gray", vmin=0, vmax=255)
    axs[0].set_title(f"Guassian (s=0.3)\nRMSE={noisy_RMSE_g03:.3f}\nPSNR={noisy_PSNR_g03:.3f}")

    axs[1].imshow(filtered_img_g03, cmap="gray", vmin=0, vmax=255)
    axs[1].set_title(f"Filtered Image\nRMSE={filtered_RMSE_g03:.3f}\nPSNR={filtered_PSNR_g03:.3f}")

    for ax in axs:
        ax.set_axis_off()

    plt.show()
    """    

    # DRID Rank Order ER vs. DRID Median against Random Impulse Noise with p=0.05 and p=0.1
    noisy_img_p005 = impulse_noise(img, noise_model="random", p=0.05)
    noisy_img_p01 = impulse_noise(img, noise_model="random", p=0.1)

    DRID_RO_filtered_img_p005 = DRID_median_rank_order_er_filter(img=noisy_img_p005, er=4)
    DRID_RO_filtered_img_p01 = DRID_median_rank_order_er_filter(img=noisy_img_p01, er=4)

    DRID_M_filtered_img_p005 = differential_rank_impulse_detection(img=noisy_img_p005)
    DRID_M_filtered_img_p01 = differential_rank_impulse_detection(img=noisy_img_p01)


    _, DRID_RO_noisy_RMSE_p005, DRID_RO_noisy_PSNR_p005 = compare_images(img, noisy_img_p005, silent=True)
    _, DRID_RO_noisy_RMSE_p01, DRID_RO_noisy_PSNR_p01 = compare_images(img, noisy_img_p01, silent=True)

    _, DRID_RO_filtered_RMSE_p005, DRID_RO_filtered_PSNR_p005 = compare_images(img, DRID_RO_filtered_img_p005, silent=True)
    _, DRID_RO_filtered_RMSE_p01, DRID_RO_filtered_PSNR_p01 = compare_images(img, DRID_RO_filtered_img_p01, silent=True)

    _, DRID_M_noisy_RMSE_p005, DRID_M_noisy_PSNR_p005 = compare_images(img, noisy_img_p005, silent=True)
    _, DRID_M_noisy_RMSE_p01, DRID_M_noisy_PSNR_p01 = compare_images(img, noisy_img_p01, silent=True)

    _, DRID_M_filtered_RMSE_p005, DRID_M_filtered_PSNR_p005 = compare_images(img, DRID_M_filtered_img_p005, silent=True)
    _, DRID_M_filtered_RMSE_p01, DRID_M_filtered_PSNR_p01 = compare_images(img, DRID_M_filtered_img_p01, silent=True)



    # Plotting of the noisy and filtered images for Random Impulse noise with p=0.05
    fig, axs = plt.subplots(1, 2)
    fig.suptitle("DRID Rank Order Filtering")
    axs[0].imshow(noisy_img_p005, cmap="gray", vmin=0, vmax=255)
    axs[0].set_title(f"Random Impulse Noise (p=0.05)\nRMSE={DRID_RO_noisy_RMSE_p005:.3f} PSNR={DRID_RO_noisy_PSNR_p005:.3f}")

    axs[1].imshow(DRID_RO_filtered_img_p005, cmap="gray", vmin=0, vmax=255)
    axs[1].set_title(f"Filtered Image\nRMSE={DRID_RO_filtered_RMSE_p005:.3f} PSNR={DRID_RO_filtered_PSNR_p005:.3f}")

    for ax in axs:
        ax.set_axis_off()

    plt.show()

    # Plotting of the noisy and filtered images for Random Impulse noise with p=0.1
    fig, axs = plt.subplots(1, 2)
    fig.suptitle("DRID Rank Order Filtering")
    axs[0].imshow(noisy_img_p01, cmap="gray", vmin=0, vmax=255)
    axs[0].set_title(f"Random Impulse Noise (p=0.1)\nRMSE={DRID_RO_noisy_RMSE_p01:.3f} PSNR={DRID_RO_noisy_PSNR_p01:.3f}")

    axs[1].imshow(DRID_RO_filtered_img_p01, cmap="gray", vmin=0, vmax=255)
    axs[1].set_title(f"Filtered Image\nRMSE={DRID_RO_filtered_RMSE_p01:.3f} PSNR={DRID_RO_filtered_PSNR_p01:.3f}")

    for ax in axs:
        ax.set_axis_off()

    plt.show()


    fig, axs = plt.subplots(1, 2)
    fig.suptitle("DRID Median Filtering")
    axs[0].imshow(noisy_img_p005, cmap="gray", vmin=0, vmax=255)
    axs[0].set_title(f"Random Impulse Noise (p=0.05)\nRMSE={DRID_M_noisy_RMSE_p005:.3f} PSNR={DRID_M_noisy_PSNR_p005:.3f}")

    axs[1].imshow(DRID_M_filtered_img_p005, cmap="gray", vmin=0, vmax=255)
    axs[1].set_title(f"Filtered Image\nRMSE={DRID_M_filtered_RMSE_p005:.3f} PSNR={DRID_M_filtered_PSNR_p005:.3f}")

    for ax in axs:
        ax.set_axis_off()

    plt.show()

    # Plotting of the noisy and filtered images for Random Impulse noise with p=0.1
    fig, axs = plt.subplots(1, 2)
    fig.suptitle("DRID Median Filtering")
    axs[0].imshow(noisy_img_p01, cmap="gray", vmin=0, vmax=255)
    axs[0].set_title(f"Random Impulse Noise (p=0.1)\nRMSE={DRID_M_noisy_RMSE_p01:.3f} PSNR={DRID_M_noisy_PSNR_p01:.3f}")

    axs[1].imshow(DRID_M_filtered_img_p01, cmap="gray", vmin=0, vmax=255)
    axs[1].set_title(f"Filtered Image\nRMSE={DRID_M_filtered_RMSE_p01:.3f} PSNR={DRID_M_filtered_PSNR_p01:.3f}")

    for ax in axs:
        ax.set_axis_off()

    plt.show()


if __name__ == "__main__":
    main()
