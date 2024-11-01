import matplotlib.pyplot as plt
import numpy as np

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
ground_end_points = np.zeros((n_lines_y_img, n_lines_x_img))
raw_depth_points = np.zeros((n_lines_y_img, n_lines_x_img))
in_roof_normalized_points = np.zeros((n_lines_y_img, n_lines_x_img))
# ground_end_point_dists = np.zeros((n_lines_y_img, n_lines_x_img))
ground_end_point_dists = []
# raw_end_points = np.zeros((n_lines_y_img, n_lines_x_img))
raw_end_points = []
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
        ground_end_points[i,j] = (gepx, gepy, gepz)
        # ground_end_point_dists[i,j] = np.sqrt(x_diff_project_x**2 + x_diff_project_y**2 + (start[2]-ground_z)**2)
        dist = np.sqrt(x_diff_project_x**2 + x_diff_project_y**2 + (start[2]-ground_z)**2)
        ground_end_point_dists.append(dist)
        raw_end_pointx = start[0]+(max_raw_depth/dist)*(gepx-start[0])
        raw_end_pointy = start[1]+(max_raw_depth/dist)*(gepy-start[1])
        raw_end_pointz = start[2]+(max_raw_depth/dist)*(gepz-start[2])
        raw_end_points[i,j] = (raw_end_pointx, raw_end_pointy, raw_end_pointz)
        r = random_values[k]
        k += 1
        # Calculate raw depth points based on random values (interpolation between start and end points)
        rdpx = start[0] + r * (raw_end_pointx - start[0])
        rdpy = start[1] + r * (raw_end_pointy - start[1])
        rdpz = start[2] + r * (raw_end_pointz - start[2])
        raw_depth_points[i,j] = (rdpx, rdpy, rdpz)
        rdp_dist = np.sqrt((rdpx-start[0])**2 + (rdpy-start[1])**2 + (rdpz-start[2])**2)
        if rdp_dist < nearest_distance:
            nearest_distance = rdp_dist
            nearest_rdp = raw_depth_points[i,j]
        if rdp_dist > farthest_distance:
            farthest_distance = rdp_dist
            farthest_point = raw_depth_points[i,j]

# Adjust irn points so the nearest point is at z=roof_z and the farthest is at z=ground_z
distance_range = farthest_distance - nearest_distance

irn_points = np.zeros((n_lines_y_img, n_lines_x_img)) # in roof normalized points
irn_dists = []

for i, angle_y in enumerate(angles_y):
    for j, angle_z in enumerate(angles_z):
        y_black = roof_z - (roof_z * (dist - nearest_distance) / distance_range)


for end, dist in zip(ground_end_points, distances):
    # Normalize the z-coordinate for the black point between z=roof_z and z=0
    
    # Interpolate the x-coordinate to stay on the line
    ratio = (y_black - start[2]) / (end[1] - start[2])  # Interpolation ratio for x
    x_black = start[0] + ratio * (end[0] - start[0])
    
    in_roof_normalized_points.append((x_black, y_black))
    inRoNoPoDistances.append(np.sqrt((x_black-start[0])**2 + (y_black-start[2])**2))

# Plot setup
plt.figure(figsize=(6, 6))
plt.xlim(-1, 25)
plt.ylim(-20, 12)

# Draw each radial line using start and end points
for end in end_points:
    plt.plot([start[0], end[0]], [start[2], end[1]], color="yellow")

for end in ground_end_points:
    plt.plot([start[0], end[0]], [start[2], end[1]], color="blue")

# Plot the red points on each blue line based on the random values
for i, point in enumerate(raw_depth_points):
    color = 'green' if i == nearest_index else 'purple' if i == farthest_index else 'red'
    plt.plot(point[0], point[1], 'o', color=color)

# Plot the black points on each line according to normalized distance placement
for black_point in in_roof_normalized_points:
    plt.plot(black_point[0], black_point[1], 'ko')  # black points

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
# Show the plot
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Radial Lines with Normalized Black Points Along Same Lines")
plt.grid(True)


def normalize(numbers):
    # Convert the input list to a NumPy array
    arr = np.array(numbers)
    
    # Calculate the minimum and maximum values of the array
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Apply min-max normalization
    normalized_arr = (arr - min_val) / (max_val - min_val)
    
    return normalized_arr


x = np.arange(len(rdpDistances))  # the label locations
width = 0.35  # the width of the bars

# Create the bar chart
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, rdpDistances, width, label='raw depth', color='red')
bars2 = ax.bar(x + width/2, inRoNoPoDistances, width, label='in-roof normalized depth', color='blue')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Values')
ax.set_title('Comparison of Two Sets of Values')
ax.set_xticks(x)
ax.legend()

plt.show()

# Print distances of the nearest and farthest points
print(f"Nearest distance: {nearest_distance}")
print(f"Farthest distance: {farthest_distance}")
