# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 09:20:05 2025

Demo-Script to convert images generated with the wellscanner to a movie

Verified to work in "opencv_test" environment
DO NOT USE IN ANY OTHER ENVIRONMENT!!! 

@author: Stelzer Lab
"""

import cv2
import os
import glob

def get_frame_number(filename):
    """
    Extract the frame number from the image filename.
    Expected filename format: date_plate_well_fXXXXX_dpt..._fd...mm.png.
    """
    base = os.path.basename(filename)
    parts = base.split('_')
    for part in parts:
        if part.startswith('f'):
            try:
                # Remove the leading 'f' and convert to integer.
                return int(part[1:])
            except ValueError:
                continue
    return None

def main():
    # Define the directory containing the images.
    image_dir = r"D:/sizeselection/gen1/20250403_plate1_well24/"

    # Retrieve a list of PNG images from the specified directory.
    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    if not image_files:
        print("No images found in the specified directory.")
        return

    # Sort image files by their frame number.
    image_files.sort(key=lambda x: get_frame_number(x))

    # Read the first image to determine frame dimensions.
    first_image = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    if first_image is None:
        print("Error reading the first image.")
        return
    height, width = first_image.shape

    # Define frames per second.
    fps = 27.0

    # Extract date, plate, and well information from the first filename.
    # Example filename: 20250402_plate1_well1_f00000_dpt-0.260_fd0.00mm.png
    base = os.path.basename(image_files[0])
    parts = base.split('_')
    date = parts[0]
    plate = parts[1]
    well = parts[2]
    output_filename = f"{date}_{plate}_{well}.mp4"  # Changed extension

    # Define the codec and create the VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed codec
    # Note: isColor is set to False for grayscale images.
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height), isColor=False)

    # Iterate through the sorted images, reading and writing them to the video.
    for file in image_files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {file}. Skipping.")
            continue
        video_writer.write(img)

    # Release the video writer object.
    video_writer.release()
    print(f"Video successfully written to {output_filename}")

if __name__ == "__main__":
    main()