import matplotlib.pyplot as plt
import numpy as np
import rotateScale

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
random_values = np.ones(n_lines_y_img)

# Calculate red points based on random values (interpolation between start and end points)
raw_depth_points = [
    (
        start[0] + r * (end[0] - start[0]),
        start[1] + r * (end[1] - start[1])
    )
    for end, r in zip(ground_end_points, random_values)
]
# raw_depth_points[5] = (raw_depth_points[5][0] + 1, raw_depth_points[5][1])
# raw_depth_points[5] = (raw_depth_points[6][0] + 1, raw_depth_points[6][1])
# raw_depth_points[7] = (raw_depth_points[7][0] + 1, raw_depth_points[6][1])
# raw_depth_points[8] = (raw_depth_points[8][0] + 1, raw_depth_points[8][1])

rdpDistances = []
for rdp in raw_depth_points:
    dist = np.sqrt((rdp[0]-start[0])**2 + (rdp[1]-start[1])**2)
    rdpDistances.append(dist)


rotateScale.plot_rotated_scaled_points(start, raw_depth_points, rdpDistances)

# Plot setup
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.figure(figsize=(6, 6))

        
for i in range(len(raw_depth_points)):
    plt.plot(raw_depth_points[i][0], rdpDistances[i], 'o', color="red")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Raw Depth point Distance from Camera Point")
plt.grid(True)

# Plot setup
plt.figure(figsize=(6, 6))
plt.xlim(-1, 25)
plt.ylim(-20, 40)

# Draw each radial line using start and end points
for end in end_points:
    plt.plot([start[0], end[0]], [start[1], end[1]], color="yellow")

for end in ground_end_points:
    plt.plot([start[0], end[0]], [start[1], end[1]], color="blue")
# Plot the red points on each blue line based on the random values
for i, point in enumerate(raw_depth_points):
    plt.plot(point[0], point[1], 'o', color="red")
for i in range(len(raw_depth_points)):
    plt.plot(raw_depth_points[i][0], rdpDistances[i], 'o', color="black")

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
# Show the plot
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Radial Lines with Normalized Black Points Along Same Lines")
plt.grid(True)
plt.show()