import cv2

# Explore edge options
def get_filtered(img, gaussian, border_type):
    return cv2.filter2D(img, -1, gaussian, borderType=border_type)


# Load an image
img = cv2.imread('../images/fall-leaves.png')
cv2.imshow('Image', img)

# TODO: Create a Gaussian filter. Use cv2.getGaussianKernel.
gaus_filter = cv2.getGaussianKernel(5, 5)
# TODO: Apply it, specifying an edge parameter (try different parameters). Use cv2.filter2D.
cv2.imshow("Constant border filtered image", get_filtered(img, gaus_filter, cv2.BORDER_CONSTANT))

cv2.imshow("Replicate border filtered image", get_filtered(img, gaus_filter, cv2.BORDER_REPLICATE))

cv2.imshow("Reflect border filtered image", get_filtered(img, gaus_filter, cv2.BORDER_REFLECT))

cv2.imshow("Reflect 101 border filtered image", get_filtered(img, gaus_filter, cv2.BORDER_REFLECT_101))

cv2.imshow("Default border filtered image", get_filtered(img, gaus_filter, cv2.BORDER_DEFAULT))

cv2.imshow("Isolated border filtered image", get_filtered(img, gaus_filter, cv2.BORDER_ISOLATED))

cv2.waitKey(0)