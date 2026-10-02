import numpy as np

def mirror(img, k=1):
    img = img.copy()

    left_edge = np.fliplr(img[:, 1:k+1])
    right_edge = np.fliplr(img[:, (-1 - k):-1])
    img = np.append(left_edge, img, axis=1)
    img = np.append(img, right_edge, axis=1)

    top_edge = np.flipud(img[1:k+1, :])
    bottom_edge = np.flipud(img[(-1 - k):-1, :])
    img = np.append(top_edge, img, axis=0)
    img = np.append(img, bottom_edge, axis=0)

    return img


"""
img = np.array(Image.open("../../DATA/images/Lena_Y.tif"))
img = mirror(img)
img = Image.fromarray(img)
img.show()

print(img[k, k])    # top left corner
print(img[k, -1-k]) # top right corner
print(img[-1-k, k]) # bottom left corner
print(img[-1-k, -1-k]) # bottom right corner
"""
