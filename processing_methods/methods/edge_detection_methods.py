import numpy as np

from methods.padding_methods import mirror

# this function implements Laplacian 1 edge detection
# take as parameters the image and the direction
def laplacian_1(img, direction="downward"):
    # 3x3 window
    filter_size = 3

    # enhanced image is same size as unpadded original image
    # cast to float64 for operations
    enhanced_img = img.astype(np.float64)
    
    # make a copy and add mirror padding to the image
    img = img.copy()
    img = mirror(img, k=(filter_size//2))

    # define the kernel for downward/upward edge detection
    if direction == "downward":
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).reshape(-1, 1)
    else:
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).reshape(-1, 1)

    # apply linear convolution
    for i in range(enhanced_img.shape[0]):
        for j in range(enhanced_img.shape[1]):
            window = img[i:filter_size+i, j:filter_size+j].reshape(1, -1)
            enhanced_img[i, j] = (window @ kernel).item()

    # return the enhanced image
    return np.clip(np.round(enhanced_img), 0, 255).astype(np.uint8)


# this function implements Laplacian 2 edge detection
# it takes as arguments the image and direction
def laplacian_2(img, direction="downward"):
    # 3x3 filter
    filter_size = 3

    # enhanced image is same size as original image
    # cast to float64 for operations
    enhanced_img = img.astype(np.float64)
    
    # copy image and add mirror padding
    img = img.copy()
    img = mirror(img, k=(filter_size//2))

    # define the kernel for downward/upward edge detection
    if direction == "downward":
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]).reshape(-1, 1)
    else:
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).reshape(-1, 1)


    # apply linear convolution
    for i in range(enhanced_img.shape[0]):
        for j in range(enhanced_img.shape[1]):
            window = img[i:filter_size+i, j:filter_size+j].reshape(1, -1)
            enhanced_img[i, j] = (window @ kernel).item()

    # return the enhanced image
    return np.clip(np.round(enhanced_img), 0, 255).astype(np.uint8)
