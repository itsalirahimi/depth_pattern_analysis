from GroundTruth2D import GroundTruth2D
import numpy as np
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
    dist = np.zeros(camera.num_of_pix)
    # Find intersections
    intersections = camera.find_intersections()
    for i, data in enumerate(intersections):
        a,b,c = data
        dist[i] = c
    camera.normalize_and_map(dist)
    # Plot the results
    camera.plot()