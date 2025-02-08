import cv2
import argparse

# Global variables
positive_points = []  # List to store positive (green) points
negative_points = []  # List to store negative (red) points
point_id = 0
img = None


def click_event(event, x, y, flags, param):
    """Mouse callback function to capture (x, y) coordinates and draw points."""
    global point_id, img
    
    if event == cv2.EVENT_LBUTTONDOWN:  # Left click: Positive point (Green)
        print(f"Positive point id = {point_id}, Clicked at [{x}, {y}]")
        positive_points.append([x, y])
        color = (0, 255, 0)  # Green
    
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right click: Negative point (Red)
        print(f"Negative point id = {point_id}, Clicked at [{x}, {y}]")
        negative_points.append([x, y])
        color = (0, 0, 255)  # Red
    
    else:
        return
    
    # Draw circle and point ID
    cv2.circle(img, (x, y), 5, color, -1)
    cv2.putText(img, str(point_id), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    point_id += 1
    cv2.imshow('Image', img)


def reset_points(image_path):
    """Resets points and reloads the original image."""
    global positive_points, negative_points, point_id, img
    
    # Print recorded points
    print ("points = [", end="")
    for p in positive_points:
        print (p, end="")
        print (", ", end="")
    for p in negative_points:
        print (p, end="")
        print (", ", end="")
    print ("]")
    print ("binaries =[", end="")
    for p in positive_points:
        print ("1, ", end="")
    for p in negative_points:
        print ("0, ", end="")
    print ("]")

    # Reload the original image
    img = cv2.imread(image_path)
    positive_points.clear()
    negative_points.clear()
    point_id = 0
    cv2.imshow('Image', img)

def main():
    global img
    
    # Argument parser
    parser = argparse.ArgumentParser(description="Mouse click event on an image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    args = parser.parse_args()
    
    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print("Error: Unable to load image.")
        return
    
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', click_event)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):  # Reset points
            reset_points(args.image)
        elif key == 27:  # ESC key to exit
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
