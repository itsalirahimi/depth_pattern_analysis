import argparse
import cv2
import numpy as np

# Argument parser
parser = argparse.ArgumentParser(description="Mouse click event on an image.")
parser.add_argument("--image", required=True, help="Path to the input image.")
args = parser.parse_args()
image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(image, cv2.CV_64F)

laplacian_norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)

# Convert back to uint8 for display
laplacian_norm = np.uint8(laplacian_norm)

threshold_value = 50  # Adjust this value as needed
_, thresholded = cv2.threshold(laplacian, threshold_value, 
                               255, cv2.THRESH_BINARY)


# Step 4: Show the result
cv2.imshow("Thresholded Laplacian Output", thresholded)
cv2.waitKey(0)
thresholded = np.uint8(thresholded)
# Apply Median Filter to remove salt-and-pepper noise
median_filtered = cv2.medianBlur(thresholded, 3)
# Step 4: Show the result
cv2.imshow("Laplacian Laplacian Output", median_filtered)
cv2.waitKey(0)
# # Apply Morphological Opening to remove small noise
# kernel = np.ones((3, 3), np.uint8)
# cleaned_output = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel)

# # Display the final result
# cv2.imshow("Final Denoised Output", cleaned_output)
# cv2.waitKey(0)
cv2.destroyAllWindows()