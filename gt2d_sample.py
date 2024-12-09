import numpy as np
import matplotlib.pyplot as plt
from GroundTruth2D import GroundTruth2D

# Script to use the class
if __name__ == "__main__":
    # Camera parameters
    camera_position = (0, 9)
    theta = -0.783653  # Central angle in radians
    fov_deg = 50       # Field of view in degrees
    num_of_pix = 330

    # Create the camera radial lines object
    gt2d = GroundTruth2D(camera_position, theta, fov_deg, num_of_pix)
    depth_data = gt2d.get_depth_data()
    depth_data = np.array(depth_data, dtype=float)
    print (depth_data)
    gt2d.plot()
