import matplotlib.pyplot as plt
import numpy as np

def rotate_and_scale_points(start_point, target_points, distances):
    """
    Rotates and scales distances around start_point to match the target points.
    
    Parameters:
    - start_point: Tuple (x, y) representing the origin point for rotations.
    - target_points: List of tuples representing the red points to match.
    - distances: List of radial distances for the black points.
    
    Returns:
    - transformed_points: List of tuples representing scaled and rotated points.
    """
    transformed_points = []
    for target, dist in zip(target_points, distances):
        # Calculate the angle to each target point from the start point
        angle = np.arctan2(target[1] - start_point[1], target[0] - start_point[0])
        # Calculate new x, y based on distance and angle
        new_x = start_point[0] + dist * np.cos(angle)
        new_y = start_point[1] + dist * np.sin(angle)
        transformed_points.append((new_x, new_y))
    return transformed_points

def plot_rotated_scaled_points(start_point, raw_depth_points, rdpDistances, colors=('red', 'black')):
    """
    Plots the target red points and the rotated-scaled black points.
    
    Parameters:
    - start_point: Tuple (x, y) representing the origin point for rotations.
    - raw_depth_points: List of tuples representing the red points.
    - rdpDistances: List of radial distances for the black points.
    - colors: Tuple with colors for red and black points respectively.
    """
    # Transform distances to get the rotated and scaled black points
    transformed_points = rotate_and_scale_points(start_point, raw_depth_points, rdpDistances)
    
    # Plotting setup
    plt.figure(figsize=(6, 6))
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Rotated and Scaled Black Points to Match Red Points")
    plt.grid(True)
    
    # Plot the red target points
    plt.plot([pt[0] for pt in raw_depth_points], [pt[1] for pt in raw_depth_points], 'o', 
             color=colors[0], label="Red Points")
    # Plot the transformed black points
    plt.plot([pt[0] for pt in transformed_points], [pt[1] for pt in transformed_points], 'o', 
             color=colors[1], label="Black Points (Transformed)")

    plt.legend()
    plt.show()
