import cv2
img = cv2.imread('../images/fruit.png')
cv2.imshow("Image", img)

cv2.imshow("Fruit image", img)

print(img.shape)

cv2.imshow("Fruit image first channel",  img[:,:,0])
cv2.imshow("Fruit image second channel", img[:,:,1])
cv2.imshow("Fruit image third channel",  img[:,:,2])

cv2.waitKey(0)