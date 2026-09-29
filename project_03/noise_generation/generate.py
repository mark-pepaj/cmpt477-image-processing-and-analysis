import numpy as np
import random
import math
from PIL import Image

# additive gaussian noise
def box_muller(m, sigma):
    r = random.random()
    fi = random.random()
    z1 = np.sqrt(-2 * np.log(r)) * np.sin(2 * math.pi * fi)
    z2 = np.sqrt(-2 * np.log(r)) * np.cos(2 * math.pi * fi)
    
    x1 = m + z1 * sigma
    x2 = m + z2 * sigma

    return x1, x2


def generate_gaussian_noise(height, width, mean, std):
    # since Box-Muller calculates two pixels at once we can generate two pixels of the noisy image at a time.
    # but if the width of the original image is not even, then there will be an error
    # after we reach the last column of the image.
    # so we can add a column if the image has an odd length: width + (width % 2)
    noise = np.zeros((height, width + (width % 2)))

    # then for every 2 pixels in a row we can assign them to x1, x2 from the Box-Muller algorithm
    for i in range(height):
        for j in range(0, width + (width % 2), 2):
            noise[i, j], noise[i, j+1] = box_muller(mean, std)
    
    # then return all the rows of the image and all the columns up to width.
    # for a 512x513 image, width + (width % 2) = 514
    # and [:width] is all the columns up to and including 513 but not including 514.
    # for a 512x512 image, width + (width % 2) = 512, so [:width] is all the columns up to and including 512
    return noise[:, :width]


# function that calls generate_guassian_noise on an image and applies the noise to the image
# it takes as arguments the mean and std of the image
# note that the std should be the fraction of the clean image's standard deviation, ex. gaussian_noise(img, mean=mean, std=0.2 * std)
def gaussian_noise(img, mean, std):
    # make copy of the image not to mutate original image and cast to float64
    img = img.copy().astype(np.float64)

    # generate a noise model
    noise = generate_gaussian_noise(height=img.shape[0], width=img.shape[1], mean=mean, std=std)
    
    # apply the noise model, round and clip the image on [0, 255] and cast to type uint8
    return np.clip(np.round(img + noise - mean), 0, 255).astype(np.uint8)
    

# this function takes the image and the percentage of pixels that will be effected by the noise
def salt_pepper_impulse_noise(img, p=0.01):
    # copy the image to not mutate original image
    img = img.copy()
    # calculate the number of affected pixels
    affected_pixels = int(np.round(p * (img.shape[0] * img.shape[1])))
    
    for i in range(affected_pixels):
        # generate random x and y coordinates
        x, y = random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1)
        # randomly select a pixel value, either 0 or 255
        pixel_value = random.randint(0, 1)
        if pixel_value == 0:
            img[x, y] = 0
        else:
            img[x, y] = 255
            
    return img


# bipolar impulse noisy works the same but it is given n1 and n2, which are the two options for a noisy pixel
def bipolar_impulse_noise(img, p=0.01, n1=30, n2=225):
    img = img.copy()
    affected_pixels = int(np.round(p * (img.shape[0] * img.shape[1])))
    
    for i in range(affected_pixels):
        x, y = random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1)
        pixel_value = random.randint(0, 1)
        if pixel_value == 0:
            img[x, y] = n1
        else:
            img[x, y] = n2
            
    return img


# random impulse noise is generated in the same way but noisy pixels are randomly generated on [n_min, n_max]
def random_impulse_noise(img, p=0.01, n_min=0, n_max=255):
    img = img.copy()
    affected_pixels = int(np.round(p * (img.shape[0] * img.shape[1])))

    for i in range(affected_pixels):
        x, y = random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1)
        pixel_value = random.randint(n_min, n_max)
        img[x, y] = pixel_value 

    return img


def impulse_noise(img, noise_model="salt and pepper", p=0.01, n1=30, n2=225, n_min=0, n_max=255):
    if noise_model == "salt and pepper":
        return salt_pepper_impulse_noise(img, p=p)
    if noise_model == "bipolar":
        return bipolar_impulse_noise(img, p=p, n1=n1, n2=n2)
    if noise_model == "random": 
        return random_impulse_noise(img, p=p, n_min=n_min, n_max=n_max)


