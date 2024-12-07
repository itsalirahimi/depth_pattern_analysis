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
    camera = GroundTruth2D(camera_position, theta, fov_deg, num_of_pix)

    # Compute radial lines and generate terrain
    camera.compute_radial_lines()
    camera.generate_simple_terrain()

    # Find intersections
    intersections = camera.find_intersections()

    print(camera.closest_points)
    print(type(camera.closest_points))
    # Plot the results
    camera.plot()
