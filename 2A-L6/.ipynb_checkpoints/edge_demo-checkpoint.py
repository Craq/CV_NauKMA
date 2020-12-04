import cv2

import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D


def surf(data: np.array):
    y = np.arange(0, data.shape[0])
    x = np.arange(0, data.shape[1])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(X, Y, data, rstride=1, cstride=1, linewidth=0,
                    cmap='jet', antialiased=False)

    plt.show(block=False)


def LoG(size: int, sigma: float) -> float:
    x = y = np.linspace(-size, size, 2*size+1)
    x, y = np.meshgrid(x, y)

    f = (x**2 + y**2)/(2*sigma**2)
    k = -1./(np.pi * sigma**4) * (1 - f) * np.exp(-f)

    return k

# Edge demo

# Read Lena image
lenaL = cv2.imread('../images/lena.png')
cv2.imshow('Original image, color', lenaL)

# Convert to monochrome (grayscale) using BGR2GRAY.
lenaMono = cv2.cvtColor(lenaL, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original image, monochrome', lenaMono)

# Make a blurred/smoothed version. Use cv2.getGaussianKernel to get the h kernel
h = cv2.getGaussianKernel(10, 3)
h = h*h.T

print( h)

# Mimic Matlab's surf(h)
# for some reason it matplotlib crashes python. It works in jupyter, you can see it there
#surf(h)

# Use cv2.filter2D with BORDER_CONSTANT to get results similar to the Matlab demo
lenaSmooth = cv2.filter2D(lenaL, -1, h)
cv2.imshow('Smoothed image', lenaSmooth)

shift_by = 1
# Method 1: Shift left and right, and show diff image
lenaL = np.copy(lenaSmooth)  # Let's use np.copy to avoid modifying the original array
# TODO: use numpy indexing to copy and paste the array to the right position
lenaL = lenaL[:, :lenaSmooth.shape[1] - shift_by]

lenaR = np.copy(lenaSmooth)  # Let's use np.copy to avoid modifying the original array
# TODO: use numpy indexing to copy and paste the array to the right position
lenaR = lenaR[:, shift_by:]

# TODO: Subtract lenaL from lenaR. Don't forget about using the correct data type
lenaDiff = cv2.subtract(lenaR, lenaL)

# Here we shift the value range to fit [0, 255] and make sure the data type is uint8 in order to display the results.
lenaDiff = cv2.normalize(lenaDiff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
cv2.imshow('Difference between right and left shifted images', lenaDiff.astype(np.uint8))

# Method 2: Canny edge detector
# OpenCV doesn't have a function similar to edge but it does have a Canny Edge detector
# OpenCV needs you to specify low and high threshold values. While these are not the
# exactly the same as the ones used in the demo you should refer to the lines below
# as a reference on how cv2.Canny works
thresh1 = 110
thresh2 = 60

cannyEdges = cv2.Canny(lenaMono, thresh1, thresh2)  # TODO: use cv2.Canny with lenaMono and the thresholds defined above
cv2.imshow('Original edges', cannyEdges)

cannyEdges = cv2.Canny(lenaSmooth, thresh1, thresh2)  # TODO: use cv2.Canny with lenaSmooth and the thresholds defined above
cv2.imshow('Edges of smoothed image', cannyEdges)

# Method 3: Laplacian of Gaussian
h = LoG(4, 1.)
# for some reason it matplotlib crashes python. It works in jupyter, you can see it there
#surf(h)

# Let's use cv2.filter2D with the new h
logEdges = cv2.filter2D(lenaSmooth, -1, h)
logEdgesShow = cv2.normalize(logEdges, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

cv2.imshow('Laplacian image before zero crossing', logEdgesShow.astype(np.uint8))

# OpenCV doesn't have a function edge like Matlab that implements a 'log' method. This would
# have to be implemented from scratch. This may take a little more time to implement this :).

cv2.waitKey(0)