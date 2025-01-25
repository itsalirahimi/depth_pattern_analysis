import matplotlib.pyplot as plt
import numpy as np
from VisualDepthGeom import VisualDepthGeometry2D
from GroundTruth2D import GroundTruth2D

from PIL import Image
import numpy as np

# # Open the image file
# image = Image.open('00000327.jpg')

# # Convert the image to a numpy array
# image_array = np.array(image)

# # Choose the column you want (e.g., column 100)
# column_index = 100
# depth_data1 = image_array[:, column_index]

# col 69:
depth_data1 = np.array([33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 32, 32, 32, 32, 32, 32, 31, 31, 31, 31, 31, 31, 29, 29, 29, 29, 29, 29, 28, 28, 28, 28, 28, 28, 26, 26, 26, 26, 26, 26, 24, 24, 24, 24, 24, 24, 21, 21, 21, 21, 21, 21, 17, 17, 17, 17, 17, 17, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 17, 17, 17, 17, 17, 17, 21, 21, 21, 21, 21, 21, 26, 26, 26, 26, 26, 26, 30, 30, 30, 30, 30, 30, 35, 35, 35, 35, 35, 35, 39, 39, 39, 39, 39, 39, 44, 44, 44, 44, 44, 44, 49, 49, 49, 49, 49, 49, 55, 55, 55, 55, 55, 55, 60, 60, 60, 60, 60, 60, 67, 67, 67, 67, 67, 67, 73, 73, 73, 73, 73, 73, 81, 81, 81, 81, 81, 81, 87, 87, 87, 87, 87, 87, 96, 96, 96, 96, 96, 96,106,106,106,106,106,106,112,112,112,112,112,112,120,120,120,120,120,120,129,129,129,129,129,129,138,138,138,138,138,138,147,147,147,147,147,147,155,155,155,155,155,155,164,164,164,164,164,164,174,174,174,174,174,174,182,182,182,182,182,182,187,187,187,187,187,187,190,190,190,190,190,190,193,193,193,193,193,193,197,197,197,197,197,197,201,201,201,201,201,201,205,205,205,205,205,205,210,210,210,210,210,210,216,216,216,216,216,216,221,221,221,221,221,221,227,227,227,227,227,227,233,233,233,233,233,233,237,237,237,237,237,237,240,240,240,240,240,240,245,245,245,245,245,245,251,251,251,251,251,251], dtype=float)
# col 245:
# depth_data = np.array([28, 28, 28, 28, 28, 28, 27, 27, 27, 27, 27, 27, 25, 25, 25, 25, 25, 25, 24, 24, 24, 24, 24, 24, 22, 22, 22, 22, 22, 22, 19, 19, 19, 19, 19, 19, 16, 16, 16, 16, 16, 16,  8,  8,  8,  8,  8,  8,  3,  3,  3,  3,  3,  3,  5,  5,  5,  5,  5,  5,  7,  7,  7,  7,  7,  7,  9,  9,  9,  9,  9,  9, 13, 13, 13, 13, 13, 13, 16, 16, 16, 16, 16, 16, 21, 21, 21, 21, 21, 21, 25, 25, 25, 25, 25, 25, 29, 29, 29, 29, 29, 29, 33, 33, 33, 33, 33, 33, 36, 36, 36, 36, 36, 36, 40, 40, 40, 40, 40, 40, 45, 45, 45, 45, 45, 45, 49, 49, 49, 49, 49, 49, 53, 53, 53, 53, 53, 53, 58, 58, 58, 58, 58, 58, 63, 63, 63, 63, 63, 63, 68, 68, 68, 68, 68, 68, 74, 74, 74, 74, 74, 74, 80, 80, 80, 80, 80, 80, 87, 87, 87, 87, 87, 87, 93, 93, 93, 93, 93, 93,100,100,100,100,100,100,106,106,106,106,106,106,117,117,117,117,117,117,131,131,131,131,131,131,143,143,143,143,143,143,152,152,152,152,152,152,159,159,159,159,159,159,167,167,167,167,167,167,176,176,176,176,176,176,184,184,184,184,184,184,189,189,189,189,189,189,194,194,194,194,194,194,198,198,198,198,198,198,201,201,201,201,201,201,205,205,205,205,205,205,209,209,209,209,209,209,213,213,213,213,213,213,217,217,217,217,217,217,222,222,222,222,222,222,227,227,227,227,227,227,231,231,231,231,231,231,236,236,236,236,236,236,239,239,239,239,239,239,241,241,241,241,241,241,245,245,245,245,245,245], dtype=float)
# col 405:
# depth_data = np.array([14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 16, 16, 16, 16, 16, 16, 18, 18, 18, 18, 18, 18, 21, 21, 21, 21, 21, 21, 24, 24, 24, 24, 24, 24, 27, 27, 27, 27, 27, 27, 31, 31, 31, 31, 31, 31, 37, 37, 37, 37, 37, 37, 47, 47, 47, 47, 47, 47, 50, 50, 50, 50, 50, 50, 53, 53, 53, 53, 53, 53, 56, 56, 56, 56, 56, 56, 60, 60, 60, 60, 60, 60, 65, 65, 65, 65, 65, 65, 71, 71, 71, 71, 71, 71, 77, 77, 77, 77, 77, 77, 83, 83, 83, 83, 83, 83, 90, 90, 90, 90, 90, 90, 99, 99, 99, 99, 99, 99,115,115,115,115,115,115,130,130,130,130,130,130,140,140,140,140,140,140,150,150,150,150,150,150,159,159,159,159,159,159,169,169,169,169,169,169,179,179,179,179,179,179,186,186,186,186,186,186,192,192,192,192,192,192,197,197,197,197,197,197,201,201,201,201,201,201,204,204,204,204,204,204,208,208,208,208,208,208,211,211,211,211,211,211,213,213,213,213,213,213,215,215,215,215,215,215,218,218,218,218,218,218,221,221,221,221,221,221,225,225,225,225,225,225,228,228,228,228,228,228,232,232,232,232,232,232,236,236,236,236,236,236], dtype=float)
# depth_data, ground_truth = get_from_groundtruth(cam_point=(0, 9.4),
#                               cam_ver_fov_deg=50,
#                               ground_z=0,
#                               cam_angle=-0.783653, num_of_pix = 330)

cam_point = (0, 9.4)
num_of_pix = 30
cam_point = cam_point
cam_ver_fov_deg = 50
fix_roof_z = 3
ground_z = 0
cam_angle = -0.783653

def analyze(dd, ax, lab, teps=None):
    vdg2d = VisualDepthGeometry2D(cam_point=cam_point,
                                cam_ver_fov_deg=cam_ver_fov_deg,
                                fix_roof_z=fix_roof_z,
                                ground_z=ground_z,
                                cam_angle=cam_angle)

    vdg2d.registerData(dd)
    p, e = vdg2d.estimate()
    # ax.plot(e[:,0], e[:,1], color='red', label="")
    ax.plot(p[:,0], p[:,1], label="estimation for "+lab)
    if not teps is None:
        ax.plot(teps[:,0], teps[:,1], label="ground_truth for " + lab)



# Create the camera radial lines object
gt2d = GroundTruth2D(cam_point, cam_angle, cam_ver_fov_deg, num_of_pix)
# # gt2d.plot()
depth_data2 = gt2d.get_depth_data()
# real_points = gt2d.get_ground_truth()
# true_end_points = gt2d.get_seen_points()
ground_truth_points = gt2d.get_closest_points()
depth_data2 = np.array(depth_data2, dtype=float)
# print (depth_data)

# find ground truth
# gt_depth = []
# for t in true_end_points:
#     gt_depth.append(np.sqrt((t[0] - cam_point[0])**2 + (t[1] - cam_point[1])**2))

# print(p)
# angles, geps, f_n, h_n, g_n = vdg2d.getInternalData()
# angles, geps, f_n, h_n, g_n, f, h, g = vdg2d.getInternalData()

# real_points_est, rdps, geps = vdg2d.estimateRealPoints()

# f(x) + g(x) = h(x)
# f, h --> regenerateFromGroundTruth --> h
# g = vdg2d.regenerateFromGroundTruth()
# Next, we must do this: f, h' --> reg... --> h, while h' = K and K is known (how to calc?)

# Plot setup
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')

# ax.plot(angles, f_n, color='blue')
# ax.plot(angles, h_n, color='red')
# ax.plot(angles, g_n, color='green')
# plt.gca().autoscale()
plt.figure()
ax2 = plt.gca()
ax2.set_aspect('equal', adjustable='box')
analyze(depth_data1, ax2, "real image")
# analyze(depth_data2, ax2, "gt", ground_truth_points)


# for pt in true_end_points:
#     ax.scatter(pt[0], pt[1], color='blue')

# for pt in g:
#     ax.scatter(pt[0], pt[1], color='red')

# print(g)

# for end in end_points:
#     ax.plot([camera_point[0], end[0]], [camera_point[1], end[1]], color="yellow")



# for end in geps:
#     ax.plot([cam_point[0], end[0]], [cam_point[1], end[1]], color="blue")

# # # Plot the red points on each blue line based on the random values
# for point in rdps:
#     ax.plot(point[0], point[1], 'o', color="red", label='rdps')

# for i, point in enumerate(irn_points_2):
#     ax.plot(point[0], point[1], 'o', color="green", label='irn_points')

# for i, point in enumerate(tcdps):
#     ax.plot(point[0], point[1], 'o', color="pink", label='tcdps')

# for i, point in enumerate(corrected_rdp):
#     ax.plot(point[0], point[1], 'o', color="orange", label='corrected_rdp')

# for i, point in enumerate(irn_points_3):
#     ax.plot(point[0], point[1], 'o', color="brown", label='corrected_rdp irned')

# for i, point in enumerate(corrected_tcdps):
#     ax.plot(point[0], point[1], 'o', color="yellow", label='corrected_tcdps')

# for point in real_points_est:
#     ax.plot(point[0], point[1], 'o', color="purple", label='corrected_tcdps irned')

# for i, point in enumerate(tcdp):
#     plt.plot(point[0], point[1], 'o', color="purple")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
# plt.xlabel("X-axis")
# plt.ylabel("Z-axis")
plt.grid(True)
plt.show()