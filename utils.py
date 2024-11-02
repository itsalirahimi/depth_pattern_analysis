import matplotlib.pyplot as plt
import numpy as np

def get_unit_vec(vec):
    return vec / np.linalg.norm(vec)

def get_euc_dist(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)

def npimshow(data, text):
    plt.figure()
    # Normalize the array to 0-255 range
    normalized_data = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))

    # Display the array using imshow
    plt.imshow(normalized_data, cmap='gray', vmin=0, vmax=255)

    plt.colorbar()  # Adds a color scale bar
    plt.title(text)

from scipy.ndimage import zoom
def resized_imread(path, r):
    # Read the image using matplotlib
    data = plt.imread(path)  # Replace with your image file path
    # Resize the image to 1/4 of its original size using scipy.ndimage.zoom
    return zoom(data, (r, r))  # (0.25, 0.25) for spatial dimensions, 1 for color channels

def normalize_array(arr):
    """
    Normalizes a 2D NumPy array so that its values are between 0 and 1.
    The maximum value will be 1 and the minimum value will be 0.
    
    Parameters:
    arr (numpy.ndarray): 2D array to normalize

    Returns:
    numpy.ndarray: Normalized 2D array with values between 0 and 1
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def normalize_under_roof(_start, roof_z, ground_end_points, points):
    nearest_distance = 1e9
    farthest_distance = -1
    dists = np.zeros((points.shape[0], points.shape[1]))
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            dist = get_euc_dist(points[i,j,:], _start)
            dists[i,j] = dist
            if dist < nearest_distance:
                nearest_distance = dist
            if dist > farthest_distance:
                farthest_distance = dist

    distance_range = farthest_distance - nearest_distance
    irn_points = np.zeros((dists.shape[0], dists.shape[1], 3)) # in roof normalized points
    irn_dists = np.zeros(dists.shape)
    for i in range(dists.shape[0]):
        for j in range(dists.shape[0]):
            # K_saturation_to_nearest = rep_dists[i,j] / nearest_distance
            # p2 = start + K_saturation_to_nearest * (tilt_corrected_depth_points[i,j,:] - start)
            # K_saturation_to_farthest = gep_dists[i,j] / farthest_distance
            # irn_point = start + K_saturation_to_farthest * (p2 - start)
            # irn_points[i,j,:] = irn_point
            z = roof_z - (roof_z * (dists[i,j]-nearest_distance) / distance_range)
            ratio = (z - _start[2]) / (ground_end_points[i,j][2] - _start[2])  # Interpolation ratio for x
            x = _start[0] + ratio * (ground_end_points[i,j][0] - _start[0])
            y = _start[1] + ratio * (ground_end_points[i,j][1] - _start[1])
            irn_points[i,j,:] = [x,y,z]
            irn_dists[i,j] = get_euc_dist((x,y,z), _start)
    
    return irn_points, irn_dists

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        # # Normalize the z-coordinate for the irn point between z=roof_z and z=0
        # z = roof_z - (roof_z * (tcdp_dists[i,j] - nearest_distance) / distance_range)
        # # Interpolate the x-coordinate to stay on the line
        # x = start[0] + ratio * (ground_end_points[i,j][0] - start[0])
        # y = start[1] + ratio * (ground_end_points[i,j][1] - start[1])
        # irn_points[i,j,0] = x
        # irn_points[i,j,1] = y
        # irn_points[i,j,2] = z
