import cv2 as cv

img = cv.imread("00000004.png")
print(img[:,405])

cv.imshow("sd", img)
cv.waitKey()



