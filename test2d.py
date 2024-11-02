import matplotlib.pyplot as plt
import numpy as np

start = (0, 10)
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
ground_end_points = [((start[0] + (ground_z - start[1]) * np.tan(angle_y)), ground_z) for angle_y in angles_y]
end_points = []
ground_end_point_dists = [np.sqrt((start[0]-gep[0])**2 + (start[1]-gep[1])**2) for gep in ground_end_points]
for gepd, gep in zip(ground_end_point_dists, ground_end_points):
    ptx = start[0] + (max_raw_depth/gepd)*(gep[0]-start[0])
    pty = start[1] + (max_raw_depth/gepd)*(gep[1]-start[1])
    end_points.append((ptx, pty))

# Generate 8 random values between 0 and 1, each associated with a blue line
random_values = np.random.rand(n_lines_y_img)

# Calculate red points based on random values (interpolation between start and end points)
raw_depth_points = [
    (
        start[0] + r * (end[0] - start[0]),
        start[1] + r * (end[1] - start[1])
    )
    for end, r in zip(end_points, random_values)
]

rdpDistances = []
for rdp in raw_depth_points:
    dist = np.sqrt((rdp[0]-start[0])**2 + (rdp[1]-start[1])**2)
    rdpDistances.append(dist)

# Calculate distances of each red point from the start
distances = [np.sqrt((point[0] - start[0])**2 + (point[1] - start[1])**2) for point in raw_depth_points]

# Identify the nearest and the farthest points
nearest_index = np.argmin(distances)
farthest_index = np.argmax(distances)
nearest_point = raw_depth_points[nearest_index]
farthest_point = raw_depth_points[farthest_index]
nearest_distance = distances[nearest_index]
farthest_distance = distances[farthest_index]

# Adjust black points so the nearest point is at z=3 and the farthest is at z=0
distance_range = farthest_distance - nearest_distance

in_roof_normalized_points = []
inRoNoPoDistances = []
for end, dist in zip(ground_end_points, distances):
    # Normalize the z-coordinate for the black point between z=3 and z=0
    y_black = 3 - (3 * (dist - nearest_distance) / distance_range)
    
    # Interpolate the x-coordinate to stay on the line
    ratio = (y_black - start[1]) / (end[1] - start[1])  # Interpolation ratio for x
    x_black = start[0] + ratio * (end[0] - start[0])
    
    in_roof_normalized_points.append((x_black, y_black))
    inRoNoPoDistances.append(np.sqrt((x_black-start[0])**2 + (y_black-start[1])**2))

# Plot setup
plt.figure(figsize=(6, 6))
plt.xlim(-1, 25)
plt.ylim(-20, 12)

# Draw each radial line using start and end points
for end in end_points:
    plt.plot([start[0], end[0]], [start[1], end[1]], color="yellow")

for end in ground_end_points:
    plt.plot([start[0], end[0]], [start[1], end[1]], color="blue")

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
