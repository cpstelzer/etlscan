import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
IMAGE_PATH = 'C:/wellscanner/test_image2.png'  # <--- !!! REPLACE WITH YOUR IMAGE PATH !!!

# 1. For IMAQ Convolute: Define your convolution kernel
#    Kernel provided by the user:
#    [0  0  0  0  0]
#    [0 -1 -1 -1  0]
#    [0 -1 10 -1  0]
#    [0 -1 -1 -1  0]
#    [0  0  0  0  0]
CONVOLUTION_KERNEL = np.array([
    [0,  0,  0,  0,  0],
    [0, -1, -1, -1,  0],
    [0, -1, 10, -1,  0],
    [0, -1, -1, -1,  0],
    [0,  0,  0,  0,  0]
], dtype=np.float32)

# Note on kernel normalization:
# The sum of this kernel's elements is 2 (10 - 8).
# If your NI Vision IMAQ Convolute VI had a "Normalize Kernel" option enabled,
# you might need to divide this kernel by its sum:
# CONVOLUTION_KERNEL = CONVOLUTION_KERNEL / 2.0
# However, cv2.filter2D will use the kernel as is, which is often the default
# behavior unless explicit normalization is selected in NI Vision.
# We will proceed without explicit normalization here unless results differ significantly.

# 2. For IMAQ Particle Filter: Define area limits
MIN_PARTICLE_AREA = 100
MAX_PARTICLE_AREA = 50000
# --- End Configuration ---

def display_image(title, image, cmap='gray'):
    """Helper function to display images using matplotlib"""
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    # plt.show() # Removed to show all at the end

def main():
    # Load the image - assuming grayscale. If color, NI Vision might pick a plane or convert.
    # Forcing grayscale here for consistency with typical image analysis tasks.
    original_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    if original_img is None:
        print(f"Error: Could not load image from {IMAGE_PATH}")
        print("Please ensure the IMAGE_PATH is correct and the image file exists.")
        return

    current_img = original_img.copy()
    display_image("1. Original Image", current_img)

    # Step 1: IMAQ Inverse
    # Inverts pixel intensities (0 becomes 255, 255 becomes 0 for 8-bit)
    img_inverted = cv2.bitwise_not(current_img)
    current_img = img_inverted
    display_image("2. IMAQ Inverse (cv2.bitwise_not)", current_img)

    # Step 2: IMAQ Convolute
    # Applies a user-defined kernel.
    # The -1 for ddepth means the output image will have the same depth as the source.
    if CONVOLUTION_KERNEL is None:
        print("Warning: CONVOLUTION_KERNEL is not defined. Skipping convolution step.")
    else:
        print("Applying convolution with kernel:\n", CONVOLUTION_KERNEL)
        img_convoluted = cv2.filter2D(current_img, -1, CONVOLUTION_KERNEL)
        current_img = img_convoluted
        display_image("3. IMAQ Convolute (cv2.filter2D)", current_img)

    # Step 3: IMAQ AutoBThreshold (clustering, Bright Objects: 128-255)
    # Otsu's binarization is a common 'auto' threshold method that fits 'clustering'.
    # It automatically determines the optimal threshold value from the image histogram.
    # cv2.THRESH_BINARY: if pixel_value > thresh, it's set to 255 (maxval), else 0.
    # This makes bright objects white (255), fitting the "Bright Objects: 128-255" description.
    # Otsu ignores the threshold value passed (0 here) and calculates its own.
    threshold_value, img_thresholded = cv2.threshold(current_img, 0, 255,
                                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"   Otsu's calculated threshold: {threshold_value}")
    current_img = img_thresholded
    display_image("4. IMAQ AutoBThreshold (cv2.threshold with THRESH_OTSU)", current_img)

    # The following steps operate on binary images (particles)
    # Find contours, which are the boundaries of the white objects (particles)
    # cv2.RETR_EXTERNAL retrieves only the extreme outer contours.
    # cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments, leaving only their end points.
    contours, hierarchy = cv2.findContours(current_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   Found {len(contours)} initial contours/particles.")

    # Prepare output images for visualization of filtered particles
    img_after_border_reject = np.zeros_like(current_img) # Black image to draw kept particles
    img_final_particles = np.zeros_like(current_img)    # Black image for final particles

    kept_contours_after_border = []

    # Step 4: IMAQ RejectBorder
    # Removes particles touching the image border.
    rows, cols = current_img.shape
    for i, cnt in enumerate(contours):
        on_border = False
        # Check bounding box first for minor optimization, then individual points if necessary
        x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(cnt)
        if x_bb == 0 or y_bb == 0 or (x_bb + w_bb) == cols or (y_bb + h_bb) == rows:
            # If bounding box touches border, check actual contour points for more precision
            # asboundingRect can be slightly larger than the contour itself on diagonals.
            for point_outer in cnt: # cnt is a list of [[x,y]] points
                x, y = point_outer[0]
                if x == 0 or x == cols - 1 or y == 0 or y == rows - 1:
                    on_border = True
                    break
        if not on_border:
            kept_contours_after_border.append(cnt)
            cv2.drawContours(img_after_border_reject, [cnt], -1, 255, -1) # -1 for filled

    print(f"   Contours remaining after border rejection: {len(kept_contours_after_border)}")
    display_image("5. IMAQ RejectBorder (Filtered Particles)", img_after_border_reject)

    # Step 5: IMAQ Particle Filter (Area: MIN_PARTICLE_AREA - MAX_PARTICLE_AREA)
    final_kept_contours = []
    for cnt in kept_contours_after_border:
        area = cv2.contourArea(cnt)
        if MIN_PARTICLE_AREA <= area <= MAX_PARTICLE_AREA:
            final_kept_contours.append(cnt)
            cv2.drawContours(img_final_particles, [cnt], -1, 255, -1) # -1 for filled

    print(f"   Contours remaining after area filter: {len(final_kept_contours)}")
    display_image("6. IMAQ Particle Filter (Area Filtered)", img_final_particles)

    # --- Visualization of final result on original image ---
    # Create a color version of the original image to draw colored contours
    output_visualization = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_visualization, final_kept_contours, -1, (0, 255, 0), 2) # Draw green contours

    # Add text for counts
    cv2.putText(output_visualization, f"Initial Contours: {len(contours)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(output_visualization, f"After Border Reject: {len(kept_contours_after_border)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(output_visualization, f"Final Particles (Area Filtered): {len(final_kept_contours)}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    display_image("7. Final Detected Particles on Original Image", output_visualization, cmap=None) # cmap=None for BGR

    print("\nScript finished.")
    print(f"Summary:")
    print(f"  - Initial contours found: {len(contours)}")
    print(f"  - Contours after border rejection: {len(kept_contours_after_border)}")
    print(f"  - Contours after area filter ({MIN_PARTICLE_AREA}-{MAX_PARTICLE_AREA} pixels): {len(final_kept_contours)}")

    plt.show() # Show all matplotlib plots

if __name__ == '__main__':
    main()