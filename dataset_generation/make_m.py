import cv2
import os
import numpy as np
import csv
import time
from lines import calcFlatGroundDepth

# List to store all rows of pixel data
m_pixel_data = []
h_pixel_data = []
r_pixel_data = []
f_pixel_data = []
output_txt_file = "output_file_paths.txt"  # Path to save the filenames
tilt_file = "/home/hamid/viot3/pm1states/camera_states.txt"

with open(tilt_file, "r") as file:
    # Read all lines
    tlines = file.readlines()
    # print(lines[234])


# Directories containing images
directory1 = '/home/hamid/viot3/output/put'
directory2 = '/home/hamid/viot3/sam/masks'

# Function to extract the numeric part of filenames for sorting
def extract_number(filename):
    # Extract the number from the filename (e.g., '00000234' from '00000234.jpg')
    return int(''.join(filter(str.isdigit, filename)))

# Get sorted lists of filenames from both directories
files1 = sorted(
    [f for f in os.listdir(directory1) if f.endswith('.jpg')],
    key=extract_number
)
files2 = sorted(
    [f for f in os.listdir(directory2) if f.endswith('.jpg')],
    key=lambda x: extract_number(x.split("_masks_jj")[0])  # Remove the "_masks_jj" suffix before sorting
)

# Ensure both directories have the same number of images
assert len(files1) == len(files2), "Mismatch in the number of files between the two directories!"

# Define the output CSV file path
m_csv = 'mmat.csv'
h_csv = 'hmat.csv'
r_csv = 'rmat.csv'
f_csv = 'fmat.csv'

# Open the text file for appending the paths
with open(m_csv, "a") as m_file, open(h_csv, "a") as h_file, open(r_csv, "a") as r_file, \
    open(f_csv, "a") as f_file, open(output_txt_file, "w") as file:
    # Loop through the sorted filenames and process corresponding images
    for k, (file1, file2) in enumerate(zip(files1, files2)):
        # Check if the numeric part of the filenames match
        if extract_number(file1) != extract_number(file2.split("_masks_jj")[0]):
            raise ValueError(f"Mismatched filenames: {file1} and {file2}")
        
        # Full paths to the images
        if file1.endswith('.jpg') and file2.endswith('.jpg'):
            image1_path = os.path.join(directory1, file1)
            image2_path = os.path.join(directory2, file2)
            
            print(f"Processing: {image1_path} and {image2_path}")
            # Add your processing code here

            # Read the image in grayscale (assuming monochrome images)
            depth = 255 - cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
            gep_depth = calcFlatGroundDepth(np.float32(tlines[k].strip()))
            residual = depth - gep_depth
            mask = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
            if depth.shape != mask.shape:
                raise ValueError("The dimensions of depth and mask do not match.")

            ret, thresh1 = cv2.threshold(mask,230,255,cv2.THRESH_BINARY)        
            # print(np.int32(thresh1[:, 32] > 0))
            # print(thresh1[:, 32] )
            
            # cv2.imshow("ns", depth)
            # cv2.imshow("res", residual)
            # cv2.imshow("nsd", thresh1)
            # cv2.waitKey()
            # time.sleep(100)
            # Flatten the image to a 1D array (each row of the image becomes a single row in the CSV)
            # flattened_image = image.flatten()
            
            # image[:, column_index]
            # Append this flattened image data as a row
            for c in range(depth.shape[1]):
                np.savetxt(m_file, [np.int32(thresh1[:, c] > 0)], delimiter=',', fmt='%d')
                np.savetxt(h_file, [depth[:, c]], delimiter=',', fmt='%d')
                np.savetxt(r_file, [residual[:, c]], delimiter=',', fmt='%d')
                np.savetxt(f_file, [gep_depth[:, c]], delimiter=',', fmt='%d')
                file.write(f"{image1_path}, {image2_path}\n")


# # Convert the list to a numpy array for easier saving
# m_pixel_data = np.array(m_pixel_data)
# h_pixel_data = np.array(h_pixel_data)
# f_pixel_data = np.array(f_pixel_data)
# r_pixel_data = np.array(r_pixel_data)

# n_csv = 'names.csv'
# image_data

# # Save the numpy array to a CSV file
# np.savetxt(m_csv, m_pixel_data, delimiter=',', fmt='%d')
# print(f"m CSV file saved to: {m_csv}")
# np.savetxt(h_csv, h_pixel_data, delimiter=',', fmt='%d')
# print(f"h CSV file saved to: {h_csv}")
# np.savetxt(r_csv, r_pixel_data, delimiter=',', fmt='%d')
# print(f"r CSV file saved to: {r_csv}")
# np.savetxt(f_csv, f_pixel_data, delimiter=',', fmt='%d')
# print(f"f CSV file saved to: {f_csv}")
# print(f"n CSV file saved to: {n_csv}")
