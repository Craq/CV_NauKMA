import cv2
import numpy as np


# Helper function
def imnoise(img_in, method, dens):

    if method == 'salt & pepper':
        img_out = np.copy(img_in)
        r, c = img_in.shape
        x = np.random.rand(r, c)
        ids = x < dens / 2.
        img_out[ids] = 0
        ids = dens / 2. <= x
        ids &= x < dens
        img_out[ids] = 255

        return img_out

    else:
        print("Method {} not yet implemented.".format(method))
        exit()

# Apply a median filter

# Read an image
img = cv2.imread('../images/moon.png', 0)
cv2.imshow('Image', img)

# TODO: Add salt & pepper noise
img_salt_pepper = imnoise(img, 'salt & pepper', 0.1)
cv2.imshow("S&P noise img", img_salt_pepper)

# TODO: Apply a median filter. Use cv2.medianBlur
img_blur = cv2.medianBlur(img_salt_pepper, 3)
cv2.imshow("Median blur img", img_blur)
cv2.waitKey(0)