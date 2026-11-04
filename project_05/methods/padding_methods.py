import numpy as np

def pad(img, pad_width=1, mode="mirror"):
    img = img.copy()

    match mode:
        case "zero":
            zeros = np.zeros((img.shape[0], pad_width))
            img = np.append(zeros, img, axis=1)
            img = np.append(img, zeros, axis=1)

            zeros = np.zeros((pad_width, img.shape[1]))
            img = np.append(zeros, img, axis=0)
            img = np.append(img, zeros, axis=0)
        case "mirror":
            left_edge = np.fliplr(img[:, 1:pad_width+1])
            right_edge = np.fliplr(img[:, (-1 - pad_width):-1])
            img = np.append(left_edge, img, axis=1)
            img = np.append(img, right_edge, axis=1)

            top_edge = np.flipud(img[1 : pad_width + 1, :])
            bottom_edge = np.flipud(img[(-1 - pad_width) : -1, :])
            img = np.append(top_edge, img, axis=0)
            img = np.append(img, bottom_edge, axis=0)

    return img


"""
zero padding
  0 0 0 0 0
  0 1 2 3 0
  0 4 5 6 0
  0 7 8 9 0
  0 0 0 0 0
"""


"""
mirror padding
  5 4 5 6 5
  
2   1 2 3   2
5   4 5 6   5
8   7 8 9   8

  5 4 5 6 5
"""

