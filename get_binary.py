import cv2

# Load the image using OpenCV
image = cv2.imread('as.png', cv2.IMREAD_GRAYSCALE)

# Apply binary thresholding
_, binary_image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY_INV)

# Save the binary image
binary_image_path_opencv = 'binary_image_opencv.png'
cv2.imwrite(binary_image_path_opencv, binary_image)

binary_image_path_opencv
