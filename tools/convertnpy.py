import numpy as np
import os
import cv2

# Input and output directories
input_dir = './tartan_air_winter/seg_right'
output_dir = './tartan_air_winter/seg_right_sem1'

def seg2vis(segnp):
    colors = np.loadtxt('./tools/segs_rgbs.txt')
    segvis = np.zeros(segnp.shape+(3,), dtype=np.uint8)

    for k in range(256):
        mask = segnp==k
        colorind = k % len(colors)
        if np.sum(mask)>0:
            segvis[mask,:] = colors[colorind]

    return segvis

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all .npy files in the input directory
npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

# Process each .npy file
for npy_file in npy_files:
    # Load the .npy file
    data = np.load(os.path.join(input_dir, npy_file))
    
    # Convert to uint8 format (assuming the data is in the range 0-255)
    # data_uint8 = np.uint8(data)
    data_uint8=seg2vis(data)
    # Save as .jpg using OpenCV
    output_file = os.path.join(output_dir, os.path.splitext(npy_file)[0] + '.jpg')
    # data_rgb = cv2.cvtColor(data_uint8, cv2.COLOR_BGR2RGB)

# Save the image with cv2.imwrite
    # cv2.imwrite(output_file, data_rgb)
    cv2.imwrite(output_file, data_uint8)
    
    print(f"Converted '{npy_file}' to '{output_file}'")