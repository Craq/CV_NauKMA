import cv2
import numpy as np


# Blend two images
def blend(a: np.array, b: np.array, alpha: float) -> np.array:
    """ Blends to images using a weight factor.
    Args:
        a (numpy.array): Image A.
        b (numpy.array): Image B.
        alpha (float): Weight factor.

    Returns:
        numpy.array: Blended Image.
    """
    a = a.astype(np.uint16) # treat uint8 underflow and overflow
    b = b.astype(np.uint16)
    result = a + (b*alpha)
    result[result > 255] = 255
    result[result < 0] = 0
    result = result.astype(np.uint8)
    return result / 255.

dolphin = cv2.imread("../images/dolphin.png")
bicycle = cv2.imread("../images/bicycle.png")

result = blend(dolphin, bicycle, 0.75)
cv2.imshow("Combined with 0.75 weight", result)
cv2.waitKey(0)