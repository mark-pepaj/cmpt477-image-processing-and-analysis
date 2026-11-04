import numpy as np
from PIL import Image





def main():

    img = np.asarray(Image.open("../DATA/images/Lena_Y.tif"))
    img = img[::4, ::4]
    print(img.shape)
    Image.fromarray(img).show()

if __name__ == "__main__":
    main()
