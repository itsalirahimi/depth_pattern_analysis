import numpy as np
from utils import *
import time 
import cv2 as cv
# This is a 2D demo of the problem, in x-z plane
# | [z] axis
# v
# camera_point
#   \
#    \
#     \     ^
#      \     \
#       \ 
#        \     \
#         \     v
#          \
# __________\
# <-      ->^
#           |
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
        ## Normalized h_n
        # 'depth_data' is so: 255 in extremely close point and 0 in extremely far point
        # The order of data in 'depth_data' is so: from top of a pixel column in depth image to down
        self.h_n = depth_data / 255.0
        ## Geometry

        n_lines_y_img = depth_data.shape[0]
        self.angles_y = np.linspace(self.cam_pitch_angle + self.cam_ver_fov / 2, 
                            self.cam_pitch_angle - self.cam_ver_fov / 2, 
                            n_lines_y_img)
        # print(self.angles_y)

        # ground end-points
        # sdf = calc_gep(self.angles_y[4], self.ground_z, self.camera_point)
        # print(sdf)
        self.geps = [calc_gep(angle, self.ground_z, self.camera_point) for angle in self.angles_y]
        # self.geps1 = [((self.camera_point[0] + (self.ground_z - self.camera_point[1]) * np.tan(angle_y)),
        #          self.ground_z) for angle_y in self.angles_y]
        
        # print(self.geps)
        # print(self.geps1)

        gep_dists = get_euc_dists_2d(self.camera_point, self.geps)
        # max gep dist
        self.mgd = np.max(gep_dists) 

        ## Normalized f_n
        self.f_n = 1 - (gep_dists / self.mgd)

        # raw end points
        self.reps = np.array([self.camera_point + direction_from_angle_y(angle) * self.mgd for
                     angle, f_n in zip(self.angles_y, self.f_n)]).reshape((len(self.angles_y), 2))

        ## Non-normalized data
        self.h = self.h_n * self.mgd
        self.f = self.mgd - gep_dists

    
    def estimate(self):
        # g_n
        self.g_n = self.h_n - self.f_n
        self.g = self.mgd * self.g_n
        self.p = [gep + unit_direction_vector(gep, self.camera_point) * g for gep, g in
                  zip(self.geps, self.g)]
        self.p = np.array(self.p).reshape((len(self.angles_y), 2))
        self.e = np.array([self.camera_point + unit_direction_vector(self.camera_point, gep) \
                           * self.mgd * (1-2*h_n) for gep, h_n in 
                           zip(self.geps, self.h_n)]).reshape((len(self.angles_y), 2))
        # self.r = [self.camera_point + get_unit_vec_if_ang(angle) * R ...]
        return self.p, self.reps

    def getInternalData(self):
        return self.angles_y, self.geps, self.f_n, self.h_n, self.g_n

    # def registerData(self, depth_data, ground_truth=None):
    #     if not ground_truth is None:
    #         self.ground_truth = ground_truth
    #     # Selective:
    #     # depth_data = normalize_array(depth_data)
    #     depth_data /= 255.0
    #     # Number of radial lines
    #     n_lines_y_img = depth_data.shape[0]
    #     # Calculate angles_y for each line between 3.75 and 5.5 o'clock in radians
    #     self.angles_y = np.linspace(self.cam_pitch_angle + self.cam_ver_fov / 2, 
    #                         self.cam_pitch_angle - self.cam_ver_fov / 2, 
    #                         n_lines_y_img)
    #     # Start and end points for each line
    #     end_points = []
    #     for gepd, gep in zip(self.gep_dists, self.geps):
    #         ptx = self.camera_point[0] + (max_raw_depth/gepd)*(gep[0]-self.camera_point[0])
    #         pty = self.camera_point[1] + (max_raw_depth/gepd)*(gep[1]-self.camera_point[1])
    #         end_points.append((ptx, pty))
    #     # Calculate red points based on pixel values (interpolation between camera_point and end points)
    #     self.rdps = [get_relative_midpoint(self.camera_point, end, r) for end, r in 
    #                  zip(end_points, depth_data)]
    #     # self.rdp_dists = get_euc_dists_2d(self.camera_point, self.rdps)

    def regenerateFromGroundTruth(self):
        generated = []
        # g = h - f
        for k, gep in enumerate(self.gep_dists):
            generated.append(abs(gep - self.ground_truth[k]))
            # print("g: ", g)
            # print("gt: ", self.ground_truth[k])
        points = []
        # h = f + g
        for k, p in enumerate(self.geps):
            # print("cp: ", self.camera_point)
            # print("p: ", p)
            pp = p + (get_unit_vec(self.camera_point - p) * generated[k])
            # print("gep: ", p)
            # print("pp: ", pp)
            # print("gk: ", generated[k])
            # print("gd: ", self.gep_dists[k])
            # print("uvec: ", get_unit_vec(self.camera_point - p))
            assert (not np.any(np.isnan(pp)))
            points.append(pp)
        return points
        # s = perform_tilt_correction_2d(self.camera_point, self.ground_truth, self.gep_dists)
        # return normalize_under_roof_2d(self.camera_point, self.fix_roof_z, self.geps, s)

    def estimateRealPoints(self):
        # irn_points_2, irn_dists_2 = normalize_under_roof_2d(camera_point, roof_z, self.geps, self.rdps)
        tcdps = perform_tilt_correction_2d(self.camera_point, self.rdps, self.gep_dists)
        tcdp_dists = get_euc_dists_2d(self.camera_point, tcdps)
        corrected_tcdp_dists = remove_derivative_and_integrate_1d(self.angles_y, tcdp_dists, self.gep_dists)
        corrected_tcdps = get_all_from_dists(self.camera_point, self.geps, corrected_tcdp_dists)
        corrected_tcdps_irn, _ = normalize_under_roof_2d(self.camera_point, self.fix_roof_z, self.geps, corrected_tcdps)
        return corrected_tcdps_irn, self.rdps, self.geps
        # corrected_rdp_dists = remove_derivative_and_integrate(self.angles_y, self.rdp_dists, self.gep_dists)
        # corrected_rdp = get_all_from_dists(self.camera_point, self.geps, corrected_rdp_dists)
        # irn_points_3, irn_dists_3 = normalize_under_roof_2d(self.camera_point, self.fix_roof_z, self.geps, corrected_rdp)


class VisualDepthGeometry3D:
    def __init__(self, cam_point, cam_hor_fov_deg, fix_roof_z, ground_z, cam_angle, 
                 img_downscale_factor):
        self.camera_point = np.array(list(cam_point))
        self.cam_hor_fov = cam_hor_fov_deg * (np.pi / 180)
        self.fix_roof_z = fix_roof_z
        self.ground_z = ground_z
        self.cam_pitch_angle = cam_angle
        self.img_downscale_factor = img_downscale_factor
    
    def registerData(self, img_path):
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        # Get the original dimensions
        original_height, original_width = img.shape
        # Calculate the new dimensions
        new_height = original_height // 16
        new_width = original_width // 16
        # Resize the image
        self.depth_data = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA) / 255.0
        self.cam_ver_fov = (float(self.depth_data.shape[0]) / self.depth_data.shape[1]) * self.cam_hor_fov
        # Number of radial lines
        n_lines_y_img = self.depth_data.shape[0]
        n_lines_x_img = self.depth_data.shape[1]
        # Calculate angles_y for each line between 3.75 and 5.5 o'clock in radians
        self.angles_y = np.linspace(self.cam_pitch_angle + self.cam_ver_fov / 2, 
									self.cam_pitch_angle - self.cam_ver_fov / 2, 
									n_lines_y_img)
        self.angles_z = np.linspace(-self.cam_hor_fov / 2, self.cam_hor_fov / 2, n_lines_x_img)
        # Start and end points for each line
        self.geps = np.zeros((n_lines_y_img, n_lines_x_img, 3))
        self.reps = np.zeros((n_lines_y_img, n_lines_x_img, 3))
        self.rdps = np.zeros((n_lines_y_img, n_lines_x_img, 3))
        self.gep_dists = np.zeros((n_lines_y_img, n_lines_x_img))
        self.max_gep_dist = -1
        for i, angle_y in enumerate(self.angles_y):
            x_diff = (self.ground_z - self.camera_point[2]) * np.tan(angle_y)
            # x_diff_roof = (self.fix_roof_z - self.camera_point[2]) * np.tan(angle_y)
			# x_radial_tick = start[0] + x_diff
			# line_length = np.sqrt((x_diff)**2 + (ground_z - start[2])**2)
            for j, angle_z in enumerate(self.angles_z):
                x_diff_project_x = x_diff * np.cos(angle_z)
                x_diff_project_y = x_diff * np.sin(angle_z)
                self.geps[i,j,:] = [self.camera_point[0] + x_diff_project_x,
                                    self.camera_point[1] + x_diff_project_y, self.ground_z]
                self.gep_dists[i,j] = get_euc_dist_3d(self.geps[i,j,:], self.camera_point)
                if self.max_gep_dist < self.gep_dists[i,j]:
                    self.max_gep_dist = self.gep_dists[i,j]

        self.all_reps = []
        for i, angle_y in enumerate(self.angles_y):
            for j, angle_z in enumerate(self.angles_z):
                repx = self.camera_point[0] + \
                    (self.max_gep_dist/self.gep_dists[i,j])*(self.geps[i,j,0]-self.camera_point[0])
                repy = self.camera_point[1] + \
                    (self.max_gep_dist/self.gep_dists[i,j])*(self.geps[i,j,1]-self.camera_point[1])
                repz = self.camera_point[2] + \
                    (self.max_gep_dist/self.gep_dists[i,j])*(self.geps[i,j,2]-self.camera_point[2])
                # self.all_reps.append([repx, repy, repz])
                self.rdps[i,j,:] = get_relative_midpoint_3d(self.camera_point,
                                                            np.array([repx, repy, repz]),
                                                            self.depth_data[i,j])
                # print(repx, repy, repz)
                # print(self.rdps[i,j,:])
                # print(self.depth_data[i,j])
                # print(self.max_gep_dist)
                # time.sleep(1000)

    def estimateRealPoints(self):
        # irn_points_2, irn_dists_2 = normalize_under_roof_2d(camera_point, roof_z, self.geps, self.rdps)
        tcdps = perform_tilt_correction_3d(self.camera_point, self.rdps, self.max_gep_dist, 
                                           self.gep_dists)
        tcdp_dists = get_euc_dists_3d(self.camera_point, tcdps)
        corrected_tcdp_dists = remove_derivative_and_integrate_2d(self.angles_y, tcdp_dists, 
                                                               	  self.gep_dists)
        corrected_tcdps = get_all_from_dists_2d(self.camera_point, self.geps, corrected_tcdp_dists)
        corrected_tcdps_irn, _ = normalize_under_roof(self.camera_point, self.fix_roof_z, 
                                                      self.geps, corrected_tcdps)
        return corrected_tcdps_irn, self.rdps, self.geps#, self.all_reps
        # corrected_rdp_dists = remove_derivative_and_integrate(self.angles_y, self.rdp_dists, self.gep_dists)
        # corrected_rdp = get_all_from_dists(self.camera_point, self.geps, corrected_rdp_dists)
        # irn_points_3, irn_dists_3 = normalize_under_roof_2d(self.camera_point, self.fix_roof_z, self.geps, corrected_rdp)
