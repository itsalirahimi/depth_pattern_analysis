import matplotlib.pyplot as plt
import numpy as np

# New center position, lowered by 3 units
center = (0, 7)
end_y = 0

# Number of radial lines, still 8 lines between 3.5 and 5.5 o'clock
n_lines = 8
# Calculate angles for each line between 3.75 and 5.5 o'clock in radians
angles = np.linspace(3.75 * np.pi / 6, 5.5 * np.pi / 6, n_lines)

# Start and end points for each line
start_points = [(center[0], center[1]) for _ in range(n_lines)]
end_points = [((center[0] + (end_y - center[1]) * np.tan(angle)), end_y) for angle in angles]

# Generate 8 random values between 0 and 1, each associated with a blue line
random_values = np.random.rand(n_lines)

# Calculate red points based on random values (interpolation between start and end points)
red_points = [
    (
        start[0] + r * (end[0] - start[0]),
        start[1] + r * (end[1] - start[1])
    )
    for start, end, r in zip(start_points, end_points, random_values)
]

# Calculate distances of each red point from the center
distances = [np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2) for point in red_points]

# Identify the nearest and the farthest points
nearest_index = np.argmin(distances)
farthest_index = np.argmax(distances)
nearest_point = red_points[nearest_index]
farthest_point = red_points[farthest_index]
nearest_distance = distances[nearest_index]
farthest_distance = distances[farthest_index]

# Adjust black points so the nearest point is at y=3 and the farthest is at y=0
distance_range = farthest_distance - nearest_distance

black_points = []
for start, end, dist in zip(start_points, end_points, distances):
    # Normalize the y-coordinate for the black point between y=3 and y=0
    y_black = 3 - (3 * (dist - nearest_distance) / distance_range)
    
    # Interpolate the x-coordinate to stay on the line
    ratio = (y_black - start[1]) / (end[1] - start[1])  # Interpolation ratio for x
    x_black = start[0] + ratio * (end[0] - start[0])
    
    black_points.append((x_black, y_black))

# Plot setup
plt.figure(figsize=(6, 6))
plt.xlim(-1, 20)
plt.ylim(-2, 10)

# Draw each radial line using start and end points
for start, end in zip(start_points, end_points):
    plt.plot([start[0], end[0]], [start[1], end[1]], color="blue")

# Plot the red points on each blue line based on the random values
for i, point in enumerate(red_points):
    color = 'green' if i == nearest_index else 'purple' if i == farthest_index else 'red'
    plt.plot(point[0], point[1], 'o', color=color)

# Plot the black points on each line according to normalized distance placement
for black_point in black_points:
    plt.plot(black_point[0], black_point[1], 'ko')  # black points

# Show the plot
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Radial Lines with Normalized Black Points Along Same Lines")
plt.grid(True)
plt.show()

# Print distances of the nearest and farthest points
print(f"Nearest distance: {nearest_distance}")
print(f"Farthest distance: {farthest_distance}")
