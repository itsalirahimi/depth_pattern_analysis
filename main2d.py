import matplotlib.pyplot as plt
import numpy as np
import rotateScale
from utils import *

ground_z = 0
roof_z = 3
cam_ver_fov = 50 * (np.pi / 180)

# This is a 2D demo of the problem, in x-z plane
# ^ [z] axis
# |
# camera_point
#   \
#    \
#     \     ^
#      \     \
#       \ 
#        \     \
#         \     v
#          \
#  _________\
#  <-     -> ^
#            |
#      
# --------------------------> [x] axis
# [y] is pointing through outside


# col 69:
# depth_data = np.array([33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 32, 32, 32, 32, 32, 32, 31, 31, 31, 31, 31, 31, 29, 29, 29, 29, 29, 29, 28, 28, 28, 28, 28, 28, 26, 26, 26, 26, 26, 26, 24, 24, 24, 24, 24, 24, 21, 21, 21, 21, 21, 21, 17, 17, 17, 17, 17, 17, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 17, 17, 17, 17, 17, 17, 21, 21, 21, 21, 21, 21, 26, 26, 26, 26, 26, 26, 30, 30, 30, 30, 30, 30, 35, 35, 35, 35, 35, 35, 39, 39, 39, 39, 39, 39, 44, 44, 44, 44, 44, 44, 49, 49, 49, 49, 49, 49, 55, 55, 55, 55, 55, 55, 60, 60, 60, 60, 60, 60, 67, 67, 67, 67, 67, 67, 73, 73, 73, 73, 73, 73, 81, 81, 81, 81, 81, 81, 87, 87, 87, 87, 87, 87, 96, 96, 96, 96, 96, 96,106,106,106,106,106,106,112,112,112,112,112,112,120,120,120,120,120,120,129,129,129,129,129,129,138,138,138,138,138,138,147,147,147,147,147,147,155,155,155,155,155,155,164,164,164,164,164,164,174,174,174,174,174,174,182,182,182,182,182,182,187,187,187,187,187,187,190,190,190,190,190,190,193,193,193,193,193,193,197,197,197,197,197,197,201,201,201,201,201,201,205,205,205,205,205,205,210,210,210,210,210,210,216,216,216,216,216,216,221,221,221,221,221,221,227,227,227,227,227,227,233,233,233,233,233,233,237,237,237,237,237,237,240,240,240,240,240,240,245,245,245,245,245,245,251,251,251,251,251,251], dtype=float)
# col 245:
# depth_data = np.array([28, 28, 28, 28, 28, 28, 27, 27, 27, 27, 27, 27, 25, 25, 25, 25, 25, 25, 24, 24, 24, 24, 24, 24, 22, 22, 22, 22, 22, 22, 19, 19, 19, 19, 19, 19, 16, 16, 16, 16, 16, 16,  8,  8,  8,  8,  8,  8,  3,  3,  3,  3,  3,  3,  5,  5,  5,  5,  5,  5,  7,  7,  7,  7,  7,  7,  9,  9,  9,  9,  9,  9, 13, 13, 13, 13, 13, 13, 16, 16, 16, 16, 16, 16, 21, 21, 21, 21, 21, 21, 25, 25, 25, 25, 25, 25, 29, 29, 29, 29, 29, 29, 33, 33, 33, 33, 33, 33, 36, 36, 36, 36, 36, 36, 40, 40, 40, 40, 40, 40, 45, 45, 45, 45, 45, 45, 49, 49, 49, 49, 49, 49, 53, 53, 53, 53, 53, 53, 58, 58, 58, 58, 58, 58, 63, 63, 63, 63, 63, 63, 68, 68, 68, 68, 68, 68, 74, 74, 74, 74, 74, 74, 80, 80, 80, 80, 80, 80, 87, 87, 87, 87, 87, 87, 93, 93, 93, 93, 93, 93,100,100,100,100,100,100,106,106,106,106,106,106,117,117,117,117,117,117,131,131,131,131,131,131,143,143,143,143,143,143,152,152,152,152,152,152,159,159,159,159,159,159,167,167,167,167,167,167,176,176,176,176,176,176,184,184,184,184,184,184,189,189,189,189,189,189,194,194,194,194,194,194,198,198,198,198,198,198,201,201,201,201,201,201,205,205,205,205,205,205,209,209,209,209,209,209,213,213,213,213,213,213,217,217,217,217,217,217,222,222,222,222,222,222,227,227,227,227,227,227,231,231,231,231,231,231,236,236,236,236,236,236,239,239,239,239,239,239,241,241,241,241,241,241,245,245,245,245,245,245], dtype=float)
# col 405:
depth_data = np.array([14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 16, 16, 16, 16, 16, 16, 18, 18, 18, 18, 18, 18, 21, 21, 21, 21, 21, 21, 24, 24, 24, 24, 24, 24, 27, 27, 27, 27, 27, 27, 31, 31, 31, 31, 31, 31, 37, 37, 37, 37, 37, 37, 47, 47, 47, 47, 47, 47, 50, 50, 50, 50, 50, 50, 53, 53, 53, 53, 53, 53, 56, 56, 56, 56, 56, 56, 60, 60, 60, 60, 60, 60, 65, 65, 65, 65, 65, 65, 71, 71, 71, 71, 71, 71, 77, 77, 77, 77, 77, 77, 83, 83, 83, 83, 83, 83, 90, 90, 90, 90, 90, 90, 99, 99, 99, 99, 99, 99,115,115,115,115,115,115,130,130,130,130,130,130,140,140,140,140,140,140,150,150,150,150,150,150,159,159,159,159,159,159,169,169,169,169,169,169,179,179,179,179,179,179,186,186,186,186,186,186,192,192,192,192,192,192,197,197,197,197,197,197,201,201,201,201,201,201,204,204,204,204,204,204,208,208,208,208,208,208,211,211,211,211,211,211,213,213,213,213,213,213,215,215,215,215,215,215,218,218,218,218,218,218,221,221,221,221,221,221,225,225,225,225,225,225,228,228,228,228,228,228,232,232,232,232,232,232,236,236,236,236,236,236], dtype=float)
# depth_data_normalized = normalize_array(depth_data)
depth_data /= 255.0
camera_angle = -0.783653
camera_point = np.array([0, 9.4])

# Number of radial lines
n_lines_y_img = depth_data.shape[0]
# Calculate angles_y for each line between 3.75 and 5.5 o'clock in radians
angles_y = np.linspace(camera_angle + cam_ver_fov / 2, 
                       camera_angle - cam_ver_fov / 2, 
                       n_lines_y_img)

# Start and end points for each line
ground_end_points = [((camera_point[0] + (ground_z - camera_point[1]) * np.tan(angle_y)), ground_z) for angle_y in angles_y]
ground_end_point_dists = get_euc_dists_2d(camera_point, ground_end_points)
max_raw_depth = np.max(ground_end_point_dists)
# end_points = []
# for gepd, gep in zip(ground_end_point_dists, ground_end_points):
#     ptx = camera_point[0] + (max_raw_depth/gepd)*(gep[0]-camera_point[0])
#     pty = camera_point[1] + (max_raw_depth/gepd)*(gep[1]-camera_point[1])
#     end_points.append((ptx, pty))

# Calculate red points based on random values (interpolation between camera_point and end points)
raw_depth_points = [get_relative_midpoint(camera_point, end, r) for end, r in zip(end_points, depth_data)]

rdp_dists = get_euc_dists_2d(camera_point, raw_depth_points)

irn_points_2, irn_dists_2 = normalize_under_roof_2d(camera_point, roof_z, ground_end_points, raw_depth_points)

# drdp_dists = calc_derivative(angles_y, rdp_dists, ret_array=True)
# dgep_dists = calc_derivative(angles_y, ground_end_point_dists, ret_array=True)
# dirn_dists = calc_derivative(angles_y, irn_dists_2, ret_array=True)
# corrected_irnp_dists = remove_derivative_and_integrate(angles_y, irn_dists_2, ground_end_point_dists)
# corrected_irnp = get_all_from_dists(camera_point, ground_end_points, corrected_irnp_dists)

tcdps = perform_tilt_correction_2d(camera_point, raw_depth_points, ground_end_point_dists)
tcdp_dists = get_euc_dists_2d(camera_point, tcdps)
corrected_tdcp_dists = remove_derivative_and_integrate(angles_y, tcdp_dists, ground_end_point_dists)
corrected_tcdps = get_all_from_dists(camera_point, ground_end_points, corrected_tdcp_dists)
corrected_tcdps_irn, _ = normalize_under_roof_2d(camera_point, roof_z, ground_end_points, corrected_tcdps)

corrected_rdp_dists = remove_derivative_and_integrate(angles_y, rdp_dists, ground_end_point_dists)
corrected_rdp = get_all_from_dists(camera_point, ground_end_points, corrected_rdp_dists)
irn_points_3, irn_dists_3 = normalize_under_roof_2d(camera_point, roof_z, ground_end_points, corrected_rdp)

# irn_points_3, irn_dists_3 = normalize_under_roof_2d(camera_point, roof_z, ground_end_points, corrected_irnp)


# Plot setup
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

plt.gca().autoscale()

# tcdp = perform_tilt_correction_2d(camera_point, raw_depth_points, ground_end_point_dists)
# tcdp_dists = get_euc_dists_2d(camera_point, tcdp)

# # Draw each radial line using camera_point and end points
# for end in end_points:
#     plt.plot([camera_point[0], end[0]], [camera_point[1], end[1]], color="yellow")

for end in ground_end_points:
    ax.plot([camera_point[0], end[0]], [camera_point[1], end[1]], color="blue")

# Plot the red points on each blue line based on the random values
for i, point in enumerate(raw_depth_points):
    ax.plot(point[0], point[1], 'o', color="red", label='rdps')

for i, point in enumerate(irn_points_2):
    ax.plot(point[0], point[1], 'o', color="green", label='irn_points')

for i, point in enumerate(tcdps):
    ax.plot(point[0], point[1], 'o', color="pink", label='tcdps')

for i, point in enumerate(corrected_rdp):
    ax.plot(point[0], point[1], 'o', color="orange", label='corrected_rdp')

for i, point in enumerate(irn_points_3):
    ax.plot(point[0], point[1], 'o', color="brown", label='corrected_rdp irned')

for i, point in enumerate(corrected_tcdps):
    ax.plot(point[0], point[1], 'o', color="yellow", label='corrected_tcdps')

for i, point in enumerate(corrected_tcdps_irn):
    ax.plot(point[0], point[1], 'o', color="purple", label='corrected_tcdps irned')
# for i, point in enumerate(tcdp):
#     plt.plot(point[0], point[1], 'o', color="purple")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Radial Lines with Normalized Black Points Along Same Lines")

# plt.figure()
# ax2 = plt.gca()

# dtcdp = calc_derivative(angles_y, tcdp_dists)
# ax2.plot(angles_y[1:], drdp_dists, color='red')
# ax2.plot(angles_y[1:], dgep_dists, color='black')
# ax2.plot(angles_y[1:], dirn_dists, color='green')
# ax2.plot(angles_y[1:], corrected_dirn, color='pink')
# print(corrected_irnp)
# ax2.plot(angles_y[1:], dtcdp, ':', color='purple')

plt.grid(True)
plt.show()