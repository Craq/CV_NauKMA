import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize(img_in: np.array) -> np.array:
    img_out = np.zeros(img_in.shape)
    cv2.normalize(img_in, img_out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img_out


# Gradient Direction
def select_gdir(gmag: np.array, gdir: np.array, mag_min: int, angle_low: int, angle_high: int) -> np.array:
    grad_dir = np.logical_and(gmag >= mag_min, gdir >= angle_low)
    grad_dir = np.logical_and(grad_dir, gdir <= angle_high)
    grad_dir = (grad_dir * 255.).astype(np.uint8)
    return grad_dir

# Load and convert image to double type, range [0, 1] for convenience
img = cv2.imread('../images/octagon.png', 0) / 255.
cv2.imshow('Image', img)  # assumes [0, 1] range for double images

# Compute x, y gradients
gx = cv2.Sobel(img, -1, dx=1, dy=0)
gy = cv2.Sobel(img, -1, dx=0, dy=1)
cv2.imshow('Gx', gx)
cv2.imshow('Gy', gy)

gmag = np.sqrt(gx**2 + gy**2)

# The minus sign here is used based on how imgradient is implemented in octave
# See https://sourceforge.net/p/octave/image/ci/default/tree/inst/imgradient.m#l61
gdir = np.arctan2(-gy, gx) * 180 / np.pi
cv2.imshow('Gmag', gmag / (4 * np.sqrt(2)))
cv2.imshow('Gdir', normalize(gdir).astype(np.uint8))

# Find pixels with desired gradient direction
my_grad = select_gdir(gmag, gdir, 1, 30, 60)
cv2.imshow('My Grad', my_grad)
cv2.waitKey(0)