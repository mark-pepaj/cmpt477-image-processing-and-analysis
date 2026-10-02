import numpy as np
import random
import math

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


def gaussian_noise(img, mean, std):
    img = img.copy()
    noise = generate_gaussian_noise(height=img.shape[0], width=img.shape[1], mean=mean, std=std)
    img = np.clip(np.round(img + noise - mean), 0, 255).astype(np.uint8)
    return img
    

# impulse noise 
def salt_pepper_impulse_noise(img, p=0.01):
    img = img.copy() 
    affected_pixels = int(np.round(p * (img.shape[0] * img.shape[1])))

    #print(affected_pixels)
    
    for i in range(affected_pixels):
        x, y = random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1)
        pixel_value = random.randint(0, 1)
        if pixel_value == 0:
            img[x, y] = 0
        else:
            img[x, y] = 255
            
    return img


def bipolar_impulse_noise(img, p=0.01, n1=0, n2=255):
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


def random_impulse_noise(img, p=0.01, n_min=0, n_max=255):
    img = img.copy()
    affected_pixels = int(np.round(p * (img.shape[0] * img.shape[1])))

    for i in range(affected_pixels):
        x, y = random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1)
        pixel_value = random.randint(n_min, n_max)
        img[x, y] = pixel_value 

    return img

