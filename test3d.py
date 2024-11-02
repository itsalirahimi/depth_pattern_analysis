import matplotlib.pyplot as plt
import numpy as np

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

qdata = resized_imread('00000004.png', 0.125) 
print(qdata.shape)
npimshow(qdata, "an image")

start = (0, 0, 10)
ground_z = 0
roof_z = 3
max_raw_depth = 27

# Number of radial lines
n_lines_y_img = 12
n_lines_x_img = 16
# Calculate angles_y for each line between 3.75 and 5.5 o'clock in radians
angles_y = np.linspace(-0.75 * np.pi / 6, -2.25 * np.pi / 6, n_lines_y_img)
angles_z = np.linspace(-1 * np.pi / 6, 1 * np.pi / 6, n_lines_x_img)

# Start and end points for each line
ground_end_points = [((), start[2], ground_z) for angle_y in angles_y]

# ^ z axis
# |
# Center 
#   \
#    \ 
#     \     ^
#      \     \
#       \ line_length
#        \     \
#         \     v
#          \
#  _________\
#  <-xdiff-> ^
#            |
#      x_radial_tick
# --------------------------> x axis
#

# Generate random values between 0 and 1, each associated with a pixel
random_values = np.random.rand(n_lines_y_img*n_lines_x_img)
k = 0
ground_end_points = np.zeros((n_lines_y_img, n_lines_x_img, 3))
raw_depth_points = np.zeros((n_lines_y_img, n_lines_x_img, 3))
rdp_dists = np.zeros((n_lines_y_img, n_lines_x_img))
# ground_end_point_dists = np.zeros((n_lines_y_img, n_lines_x_img))
ground_end_point_dists = []
# raw_end_points = np.zeros((n_lines_y_img, n_lines_x_img))
raw_end_points = np.zeros((n_lines_y_img, n_lines_x_img, 3))
nearest_distance = 1e9
farthest_distance = -1
for i, angle_y in enumerate(angles_y):
    x_diff = (ground_z - start[2]) * np.tan(angle_y)
    # x_radial_tick = start[0] + x_diff
    # line_length = np.sqrt((x_diff)**2 + (ground_z - start[2])**2)
    for j, angle_z in enumerate(angles_z):
        x_diff_project_x = x_diff*np.cos(angle_z)
        x_diff_project_y = x_diff*np.sin(angle_z)
        gepx = start[0]+x_diff_project_x
        gepy = start[1]+x_diff_project_y
        gepz = ground_z
        ground_end_points[i,j,0] = gepx
        ground_end_points[i,j,1] = gepy
        ground_end_points[i,j,2] = gepz
        # ground_end_point_dists[i,j] = np.sqrt(x_diff_project_x**2 + x_diff_project_y**2 + (start[2]-ground_z)**2)
        dist = np.sqrt(x_diff_project_x**2 + x_diff_project_y**2 + (start[2]-ground_z)**2)
        ground_end_point_dists.append(dist)
        raw_end_pointx = start[0]+(max_raw_depth/dist)*(gepx-start[0])
        raw_end_pointy = start[1]+(max_raw_depth/dist)*(gepy-start[1])
        raw_end_pointz = start[2]+(max_raw_depth/dist)*(gepz-start[2])
        raw_end_points[i,j,0] = raw_end_pointx
        raw_end_points[i,j,1] = raw_end_pointy
        raw_end_points[i,j,2] = raw_end_pointz
        r = random_values[k]
        k += 1
        # Calculate raw depth points based on random values (interpolation between start and end points)
        rdpx = start[0] + r * (raw_end_pointx - start[0])
        rdpy = start[1] + r * (raw_end_pointy - start[1])
        rdpz = start[2] + r * (raw_end_pointz - start[2])
        raw_depth_points[i,j,0] = rdpx
        raw_depth_points[i,j,1] = rdpy
        raw_depth_points[i,j,2] = rdpz
        rdp_dist = np.sqrt((rdpx-start[0])**2 + (rdpy-start[1])**2 + (rdpz-start[2])**2)
        rdp_dists[i,j] = rdp_dist
        if rdp_dist < nearest_distance:
            nearest_distance = rdp_dist
            nearest_rdp = raw_depth_points[i,j]
        if rdp_dist > farthest_distance:
            farthest_distance = rdp_dist
            farthest_point = raw_depth_points[i,j]

# Adjust irn points so the nearest point is at z=roof_z and the farthest is at z=ground_z
distance_range = farthest_distance - nearest_distance

irn_points = np.zeros((n_lines_y_img, n_lines_x_img, 3)) # in roof normalized points
irn_dists = np.zeros((n_lines_y_img, n_lines_x_img)) 
for i, angle_y in enumerate(angles_y):
    for j, angle_z in enumerate(angles_z):
        # Normalize the z-coordinate for the irn point between z=roof_z and z=0
        z = roof_z - (roof_z * (rdp_dists[i,j] - nearest_distance) / distance_range)
        ratio = (z - start[2]) / (ground_end_points[i,j][2] - start[2])  # Interpolation ratio for x
        # Interpolate the x-coordinate to stay on the line
        x = start[0] + ratio * (ground_end_points[i,j][0] - start[0])
        y = start[1] + ratio * (ground_end_points[i,j][1] - start[1])
        irn_points[i,j,0] = x
        irn_points[i,j,1] = y
        irn_points[i,j,2] = z
        irn_dists[i,j] = np.sqrt((x-start[0])**2 + (y-start[1])**2 + (z-start[2])**2)

# Plot setup
# ax = fig.add_subplot(2, 1, 2, projection='3d')
# plt.ylim(-20, 12)

# from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax2 = fig.gca()

# all_rdp_xs = []
# all_rdp_ys = []
# all_rdp_zs = []
# all_irnp_xs = []
# all_irnp_ys = []
# all_irnp_zs = []
# for i in range(ground_end_points.shape[0]):
#     for j in range(ground_end_points.shape[1]):
#         ax.plot([start[0], ground_end_points[i,j,0]], [start[1], ground_end_points[i,j,1]], [start[2], ground_end_points[i,j,2]], color='blue')
#         ax.plot([start[0], raw_end_points[i,j,0]], [start[1], raw_end_points[i,j,1]], [start[2], raw_end_points[i,j,2]], color='yellow')
#         all_rdp_xs.append(raw_depth_points[i,j,0])
#         all_rdp_ys.append(raw_depth_points[i,j,1])
#         all_rdp_zs.append(raw_depth_points[i,j,2])
#         all_irnp_xs.append(irn_points[i,j,0])
#         all_irnp_ys.append(irn_points[i,j,1])
#         all_irnp_zs.append(irn_points[i,j,2])

# ax.scatter(all_rdp_xs, all_rdp_ys, all_rdp_zs, color = 'red')
# ax.scatter(all_irnp_xs, all_irnp_ys, all_irnp_zs, color = 'black')

npimshow(rdp_dists, "rdp")
npimshow(irn_dists, "irnp")

plt.show()

