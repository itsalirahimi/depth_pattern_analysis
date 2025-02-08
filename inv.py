import cv2

# Load the image using OpenCV
image1 = cv2.imread('00000044.jpg', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread('mask1.png', cv2.IMREAD_GRAYSCALE)
# image3 = cv2.imread('test_image_1.png', cv2.IMREAD_GRAYSCALE)
# image4 = cv2.imread('test_mask_1.png', cv2.IMREAD_GRAYSCALE)

# Invert the colors of the image
inv_image1 = 255 - image1
# inv_image2 = 255 - image2
# inv_image3 = 255 - image3
# inv_image4 = 255 - image4

# Save the inverted image
cv2.imwrite("dddddd-Fill_1.png", inv_image1)
# cv2.imwrite("inverted_mask1.png", inv_image2)
# cv2.imwrite("inverted_test_image_1.png", inv_image3)
# cv2.imwrite("inverted_test_mask_1.png", inv_image4)

