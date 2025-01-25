import cv2 as cv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Load the matrices from CSV files
hmat = pd.read_csv('hmat.csv', header=None)
rmat = pd.read_csv('rmat.csv', header=None)
mmat = pd.read_csv('mmat.csv', header=None)
fmat = pd.read_csv('fmat.csv', header=None)
output_txt_file = "output_file_paths.txt"


# Ensure the matrices have the same dimensions
if rmat.shape != mmat.shape:
    raise ValueError("The dimensions of rmat.csv and mmat.csv do not match.")

with open(output_txt_file, "r") as file:
    # Read all lines
    lines = file.readlines()
    # print(lines[234])

while True:
    # Randomly select a row index
    random_row_index = np.random.randint(0, rmat.shape[0])

    # Extract the selected row from rmat and mmat
    hmat_row = hmat.iloc[random_row_index]
    rmat_row = rmat.iloc[random_row_index]
    mmat_row = mmat.iloc[random_row_index]
    fmat_row = fmat.iloc[random_row_index]

    # Validate the row number
    if random_row_index < 1 or random_row_index > len(lines):
        raise ValueError(f"Row number {random_row_index} is out of range. File has {len(lines)} rows.")
    
    # Choose a random line
    random_line = lines[random_row_index].strip()
    
    # Split the line into image1_path and image2_path
    image1_path, image2_path = random_line.split(", ")
    print(image1_path, image2_path)
    # Load the image
    image1 = 255 - cv.imread(image1_path, cv.IMREAD_GRAYSCALE)
    image2 = cv.imread(image2_path)
    ret, thresh1 = cv.threshold(image2,230,255,cv.THRESH_BINARY)
    cv.line(image1, (random_row_index % image1.shape[1], 0), (random_row_index % image1.shape[1], image1.shape[0]-1), 255, 2)
    cv.line(image2, (random_row_index % image2.shape[1], 0), (random_row_index % image2.shape[1], image2.shape[0]-1), 255, 2)
    cv.imshow("depth", image1)
    cv.imshow("mask", image2)
    cv.imshow("mask thresh", thresh1)
    cv.waitKey()

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Iterate through the row to plot points based on mmat values
    for i, (h_val, r_val, m_val, f_val) in enumerate(zip(hmat_row, rmat_row, mmat_row, fmat_row)):
        colorh = 'red' if m_val == 1 else 'blue'
        colorr = 'yellow' if m_val == 1 else 'green'
        plt.scatter(i, h_val, color=colorh)
        plt.scatter(i, r_val, color=colorr)
        plt.scatter(i, f_val, color='black')

    # Label the axes and add a title
    plt.xlabel('Index (Column Number)')
    plt.ylabel('Value (rmat)')
    plt.title(f'Scattered Plot for Random Row {random_row_index}')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.show()
