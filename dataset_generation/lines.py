import numpy as np
import cv2 as cv

def calcFlatGroundDepth(tilt_angle, h=10, vertical_fov_degrees=0.75*66.0, horizontal_fov_degrees=66.0, resolution=(440, 330)):
    """
    Calculate the depth data for a camera image based on the geometry of light rays.

    Parameters:
        tilt_angle (float): The tilt angle of the camera in radians.
        h (float): Altitude of the camera in meters. Default is 10.
        vertical_fov_degrees (float): Vertical field of view in degrees. Default is 0.75*66.0.
        horizontal_fov_degrees (float): Horizontal field of view in degrees. Default is 66.0.
        resolution (tuple): Resolution of the camera (width, height). Default is (440, 330).

    Returns:
        np.ndarray: Normalized depth data as a 2D numpy array.
    """
    width, height = resolution

    # Convert FOVs to radians
    vertical_fov = np.deg2rad(vertical_fov_degrees)
    horizontal_fov = np.deg2rad(horizontal_fov_degrees)

    # Generate angles for the vertical and horizontal directions
    vertical_angles = (np.linspace(-vertical_fov / 2, vertical_fov / 2, height) + tilt_angle)[::-1]
    horizontal_angles = np.linspace(-horizontal_fov / 2, horizontal_fov / 2, width)

    # Create a 2D array to store depth values
    depth_data = np.zeros((height, width))

    # Calculate depth for each pixel
    for i, v_angle in enumerate(vertical_angles):
        for j, h_angle in enumerate(horizontal_angles):
            if -np.pi / 2 <= v_angle <= 0:  # Ensure valid vertical angles
                r = h / abs(np.sin(v_angle))  # Length in vertical plane
                R = r / abs(np.cos(h_angle))  # Adjust for horizontal angle
                depth_data[i, j] = R
            # else:
            #     depth_data[i, j] = np.inf  # Invalid pixels set to infinity

    # Normalize depth data
    # valid_depths = depth_data[np.isfinite(depth_data)]  # Filter out invalid depths
    max_depth = depth_data.max()
    # print((depth_data - min_depth) / (max_depth - min_depth))
    return (255 * (1 - depth_data / max_depth)).astype(np.uint8)


if __name__ == '__main__':
    # Example usage
    tilt_angle = -0.78

    cv.imshow("sd", calcFlatGroundDepth(tilt_angle))
    cv.waitKey()
