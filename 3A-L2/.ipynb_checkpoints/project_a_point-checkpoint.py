import cv2
import numpy as np

# Project a point from 3D to 2D using a matrix operation

# Given: Point p in 3-space [x y z], and focal length f
# Return: Location of projected point on 2D image plane [u v]


def project_point(p: np.array, f: int) -> np.array:
    projection_matrix = np.array([
        [f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, 0, 1]
    ])
    p = np.concatenate((p, np.zeros((1,1))),1).T
    return np.dot(projection_matrix, p).T[:,:2]

# Test: Given point and focal length (units: mm)
p = np.array([[200, 100, 120]])
f = 50

print (project_point(p, f))