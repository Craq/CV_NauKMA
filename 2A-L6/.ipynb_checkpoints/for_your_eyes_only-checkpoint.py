import cv2
import numpy as np


# For Your Eyes Only
frizzy = cv2.imread('../images/frizzy.png')
froomer = cv2.imread('../images/froomer.png')
cv2.imshow('Frizzy', frizzy)
cv2.imshow('Froomer', froomer)


# TODO: Find edges in frizzy and froomer images
frizzy_edge = cv2.Canny(frizzy, 110, 60)
froomer_edge = cv2.Canny(froomer, 110, 60)
# TODO: Display common edge pixels
cv2.imshow("Eyes only", (frizzy_edge & froomer_edge).astype(np.uint8))
cv2.waitKey(0)