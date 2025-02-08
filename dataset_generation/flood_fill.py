import cv2
from matplotlib import pyplot as plt
import numpy as np
from collections import deque

# Create a black background
image = cv2.imread("dexinet.png", cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
# kernel = np.ones((3, 3), np.uint8)
# binary_image = cv2.bitwise_not(binary_image)
# test_image = cv2.dilate(binary_image, kernel, iterations=2)
# test_image = cv2.bitwise_not(test_image)
test_image = binary_image

# # sample
# test_image = np.zeros((500, 500), dtype=np.uint8)
# # Draw white lines on a black background
# cv2.line(test_image, (0, 100), (500, 100), 255, 20)  # Thick horizontal line at y=100
# cv2.line(test_image, (0, 320), (500, 300), 255, 2)  # Horizontal line at y=300
# cv2.line(test_image, (0, 0), (100, 500), 255, 2)  # Diagonal line

# # Create an initial mask as a white rectangle at the bottom-left corner
# test_mask = np.ones_like(test_image) * 255  # Start with a white mask
# cv2.rectangle(test_mask, (0, 250), (20, 300), 0, -1)  # Small black area to expand

test_mask = cv2.imread("test_mask.png", cv2.IMREAD_GRAYSCALE)


def expand_mask_with_flood_fill(lines, mask):
    """
    Expands the black mask in all directions (upward, downward, leftward, rightward)
    until it reaches the nearest white line or the image boundary, using a flood-fill approach.
    """
    height, width = mask.shape
    expanded_mask = mask.copy()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    # Initialize queue with all the initial black mask positions
    queue = deque([(y, x) for y in range(height) for x in range(width) if mask[y, x] == 0])

    while queue:
        y, x = queue.popleft()

        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                if lines[ny, nx] == 0 and expanded_mask[ny, nx] == 255:  # Expand only into black areas
                    expanded_mask[ny, nx] = 0
                    queue.append((ny, nx))

    return expanded_mask

# Run the optimized flood-fill expansion
result_mask_flood_fill = expand_mask_with_flood_fill(test_image, test_mask)

# Displaying images
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Test Image with Lines")
plt.imshow(test_image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Initial Mask (Inverted)")
plt.imshow(test_mask, cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Flood-Fill Expanded Mask")
plt.imshow(result_mask_flood_fill, cmap='gray')
plt.tight_layout()
plt.show()
