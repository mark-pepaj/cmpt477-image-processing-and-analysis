import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from methods.image_statistics import *
from methods.noise_generation_methods import *
from methods.noise_filtering_methods import *

def visualize_filter_fft(W, pad_height=512, pad_width=512, save_path=None, silent=False):

    h = np.zeros((pad_height, pad_width))
    h[:W.shape[0], :W.shape[1]] = W
    H = np.fft.fft2(h)
    log_magnitude = np.log1p(np.abs(np.fft.fftshift(H)))

    plt.style.use('dark_background')
    x = np.arange(H.shape[0])
    y = np.arange(H.shape[1])
    X, Y = np.meshgrid(y, x)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    surface = ax.plot_surface(X, Y, log_magnitude, cmap="viridis")


    if save_path:
        plt.savefig(save_path)
        plt.close()

    if not silent:
        plt.show() 
        plt.close()



def main():

    img = np.asarray(Image.open("DATA/images/Lena_Y.tif"))

    # smart mean filter
    #W = np.array([
    #    [1, 2, 1],
    #    [2, 4, 2],
    #    [1, 2, 1]
    #    ]) / 16

    # improved smart mean filter
    W = np.array([
        [1, 3, 1],
        [3, 7, 3],
        [1, 3, 1]
        ]) / 23 

    noisy_img = gaussian_noise(img, mean=get_mean(img), std=0.1*get_std(img))
    Image.fromarray(noisy_img).save("report/images/noisy_Lena_Y.png")
    print("noisy image vs clean image:")
    compare_images(img, noisy_img, silent=False)

    spatial_filtered_img = linear_convolution_filter(noisy_img, W, mode="spatial")
    Image.fromarray(spatial_filtered_img).save("report/images/improved_spatial_filtered_Lena_Y.png")
    print("\nspatial domain filtering:")
    compare_images(img, spatial_filtered_img, silent=False)

    frequency_filtered_img = linear_convolution_filter(noisy_img, W, mode="frequency")     
    Image.fromarray(frequency_filtered_img).save("report/images/improved_frequency_filtered_Lena_Y.png")
    print("\nfrequency domain filtering:")                                                                
    compare_images(img, frequency_filtered_img, silent=False)
    
    #visualize_filter_fft(W=W, pad_height=img.shape[0], pad_width=img.shape[1], save_path="report/images/improved_smart_mean.png", silent=True)
    #Image.fromarray(noisy_img).show()
    #Image.fromarray(spatial_filtered_img).show()
    #Image.fromarray(noisy_img).show()                                                    
    #Image.fromarray(frequency_filtered_img).show()


    h = np.zeros((img.shape[0], img.shape[1]))
    h[:2, :2] = W[1:, 1:]
    h[0:2, -1] = W[1:, 0]
    h[-1, :2] = W[0, 1:]
    h[-1, -1] = W[0, 0] 

    H = np.fft.fft2(h)

    # top left corner of the power spectrum
    magnitude = np.abs(H)[:int(H.shape[0]/4), :int(H.shape[1]/4)]

    small = False
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            if magnitude[i, j] < 1e-6:
                print(f"magnitude is < 1e-6 at index ({i},{j})")
                small = True
                break
        if small:
            break

if __name__ == "__main__":
    main()

    #W = np.ones((window_size, window_size)) * window_size**-2

    # downward
    #W = np.array([
    #    [1, 1, 1],
    #    [1, -8, 1],
    #    [1, 1, 1]
    #    ])

    # upward
    #W = np.array([
    #    [-1, -1, -1],
    #    [-1,  8, -1],
    #    [-1, -1, -1]
    #    ])


#print(f)
#log_magnitude = np.log1p(np.abs(np.fft.fftshift(H)))

#x = np.arange(H.shape[0])
#y = np.arange(H.shape[1])
#X, Y = np.meshgrid(x, y)

#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(projection="3d")
#surface = ax.plot_surface(X, Y, log_magnitude, cmap="viridis")
#plt.show() 
#plt.savefig("graphs/laplacian2_upward_lena.png")
#plt.close()

