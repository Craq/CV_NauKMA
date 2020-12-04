import cv2
import numpy as np

def downsample(img: np.array) -> np.array:
    return img[np.arange(1, img.shape[0], 2)]


def blur_downsample(img: np.array) -> np.array:
    h = cv2.getGaussianKernel(3, 1)
    h = h*h.T
    gauss = cv2.filter2D(img, -1, h)
    downsampled = downsample(gauss)
    return downsampled


img = cv2.imread('../images/frizzy.png')
cv2.imshow("original_image", img)
print(img.shape)

# downsample image
img_d = downsample(img)
img_d = downsample(img_d)
img_d = downsample(img_d)
print(img_d.shape)

# blur and downsample
img_bd = blur_downsample(img)
img_bd = blur_downsample(img_bd)
img_bd = blur_downsample(img_bd)
print(img_bd.shape)

cv2.imshow("downsampled_image", cv2.resize(img_d, (img.shape[1], img.shape[0])))
cv2.imshow("blur_downsampled_image", cv2.resize(img_bd, (img.shape[1], img.shape[0])))
cv2.waitKey(0)
