import cv2
import numpy as np

# Apply a Gaussian filter to remove noise
img = cv2.imread('../images/saturn.png')
cv2.imshow('Img', img)

noise = np.random.choice([0, 255], size=img.shape, p=[0.95,0.05]).astype(np.uint8)
noise_img = cv2.add(img, noise)
cv2.imshow('Img with noise', noise_img)

blur = cv2.GaussianBlur(noise_img, (19,19), 0)
cv2.imshow('Denoised img', blur)
cv2.waitKey(0)