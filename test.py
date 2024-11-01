import matplotlib.pyplot as plt
import numpy as np

start = (0, 10)
end_z = 0

# Number of radial lines
n_lines = 8
# Calculate angles_x for each line between 3.75 and 5.5 o'clock in radians
angles_x = np.linspace(3.75 * np.pi / 6, 5.5 * np.pi / 6, n_lines)
angles_y = np.linspace(3.75 * np.pi / 6, 5.5 * np.pi / 6, n_lines)

# Start and end points for each line
end_points = [((start[0] + (end_z - start[1]) * np.tan(angle_x)), end_z) for angle_x, angles_y in zip(angles_x, angles_y)]

# Generate 8 random values between 0 and 1, each associated with a blue line
random_values = np.random.rand(n_lines)

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
    print("rdp data: ", rdp, dist)

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
for end, dist in zip(end_points, distances):
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
plt.ylim(-2, 12)

# Draw each radial line using start and end points
for end in end_points:
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
