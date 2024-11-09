import numpy as np
from utils import *

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

class VisualDepthGeometry2D:
    def __init__(self, cam_point, cam_ver_fov_deg, fix_roof_z, ground_z, cam_angle):
        self.camera_point = np.array(list(cam_point))
        self.cam_ver_fov = cam_ver_fov_deg * (np.pi / 180)
        self.fix_roof_z = fix_roof_z
        self.ground_z = ground_z
        self.cam_pitch_angle = cam_angle
    
    def registerData(self, depth_data):
        # Selective:
        # depth_data = normalize_array(depth_data)
        depth_data /= 255.0
        # Number of radial lines
        n_lines_y_img = depth_data.shape[0]
        # Calculate angles_y for each line between 3.75 and 5.5 o'clock in radians
        self.angles_y = np.linspace(self.cam_pitch_angle + self.cam_ver_fov / 2, 
                            self.cam_pitch_angle - self.cam_ver_fov / 2, 
                            n_lines_y_img)
        # Start and end points for each line
        self.geps = [((self.camera_point[0] + (self.ground_z - self.camera_point[1]) * np.tan(angle_y)), 
                      self.ground_z) for angle_y in self.angles_y]
        self.gep_dists = get_euc_dists_2d(self.camera_point, self.geps)
        max_raw_depth = np.max(self.gep_dists)

        end_points = []
        for gepd, gep in zip(self.gep_dists, self.geps):
            ptx = self.camera_point[0] + (max_raw_depth/gepd)*(gep[0]-self.camera_point[0])
            pty = self.camera_point[1] + (max_raw_depth/gepd)*(gep[1]-self.camera_point[1])
            end_points.append((ptx, pty))

        # Calculate red points based on pixel values (interpolation between camera_point and end points)
        self.rdps = [get_relative_midpoint(self.camera_point, end, r) for end, r in 
                     zip(end_points, depth_data)]
        self.rdp_dists = get_euc_dists_2d(self.camera_point, self.rdps)

    def estimateRealPoints(self):
        # irn_points_2, irn_dists_2 = normalize_under_roof_2d(camera_point, roof_z, self.geps, self.rdps)
        tcdps = perform_tilt_correction_2d(self.camera_point, self.rdps, self.gep_dists)
        tcdp_dists = get_euc_dists_2d(self.camera_point, tcdps)
        corrected_tdcp_dists = remove_derivative_and_integrate(self.angles_y, tcdp_dists, self.gep_dists)
        corrected_tcdps = get_all_from_dists(self.camera_point, self.geps, corrected_tdcp_dists)
        corrected_tcdps_irn, _ = normalize_under_roof_2d(self.camera_point, self.fix_roof_z, self.geps, corrected_tcdps)
        return corrected_tcdps_irn, self.rdps, self.geps
        # corrected_rdp_dists = remove_derivative_and_integrate(self.angles_y, self.rdp_dists, self.gep_dists)
        # corrected_rdp = get_all_from_dists(self.camera_point, self.geps, corrected_rdp_dists)
        # irn_points_3, irn_dists_3 = normalize_under_roof_2d(self.camera_point, self.fix_roof_z, self.geps, corrected_rdp)


class VisualDepthGeometry3D:
    def __init__(self, cam_point, cam_ver_fov_deg, fix_roof_z, ground_z, cam_angle):
        self.camera_point = np.array(list(cam_point))
        self.cam_ver_fov = cam_ver_fov_deg * (np.pi / 180)
        self.cam_hor_fov = cam_ver_fov_deg * (np.pi / 180)
        self.fix_roof_z = fix_roof_z
        self.ground_z = ground_z
        self.cam_pitch_angle = cam_angle
    
    def registerData(self, depth_data):
        # Selective:
        # depth_data = normalize_array(depth_data)
        depth_data /= 255.0
        # Number of radial lines
        n_lines_y_img = depth_data.shape[0]
        n_lines_x_img = depth_data.shape[1]
        # Calculate angles_y for each line between 3.75 and 5.5 o'clock in radians
        self.angles_y = np.linspace(self.cam_pitch_angle + self.cam_ver_fov / 2, 
									self.cam_pitch_angle - self.cam_ver_fov / 2, 
									n_lines_y_img)
        self.angles_x = np.linspace(self.cam_hor_fov / 2, -self.cam_hor_fov / 2, 
									n_lines_x_img)
		
        # Start and end points for each line
        self.geps = [((self.camera_point[0] + (self.ground_z - self.camera_point[1]) * np.tan(angle_y)), 
                      self.ground_z) for angle_y in self.angles_y]
        self.gep_dists = get_euc_dists_2d(self.camera_point, self.geps)
        max_raw_depth = np.max(self.gep_dists)

        end_points = []
        for gepd, gep in zip(self.gep_dists, self.geps):
            ptx = self.camera_point[0] + (max_raw_depth/gepd)*(gep[0]-self.camera_point[0])
            pty = self.camera_point[1] + (max_raw_depth/gepd)*(gep[1]-self.camera_point[1])
            end_points.append((ptx, pty))

        # Calculate red points based on pixel values (interpolation between camera_point and end points)
        self.rdps = [get_relative_midpoint(self.camera_point, end, r) for end, r in 
                     zip(end_points, depth_data)]
        self.rdp_dists = get_euc_dists_2d(self.camera_point, self.rdps)

    def estimateRealPoints(self):
        # irn_points_2, irn_dists_2 = normalize_under_roof_2d(camera_point, roof_z, self.geps, self.rdps)
        tcdps = perform_tilt_correction_2d(self.camera_point, self.rdps, self.gep_dists)
        tcdp_dists = get_euc_dists_2d(self.camera_point, tcdps)
        corrected_tdcp_dists = remove_derivative_and_integrate(self.angles_y, tcdp_dists, self.gep_dists)
        corrected_tcdps = get_all_from_dists(self.camera_point, self.geps, corrected_tdcp_dists)
        corrected_tcdps_irn, _ = normalize_under_roof_2d(self.camera_point, self.fix_roof_z, self.geps, corrected_tcdps)
        return corrected_tcdps_irn, self.rdps, self.geps
        # corrected_rdp_dists = remove_derivative_and_integrate(self.angles_y, self.rdp_dists, self.gep_dists)
        # corrected_rdp = get_all_from_dists(self.camera_point, self.geps, corrected_rdp_dists)
        # irn_points_3, irn_dists_3 = normalize_under_roof_2d(self.camera_point, self.fix_roof_z, self.geps, corrected_rdp)
