# src/visualization/stage_04_visualization.py
import skimage.io
import skimage.color
import skimage.util
import skimage.measure 
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Dict, Any, List, Tuple
from skimage.measure._regionprops import RegionProperties 
import logging

logger = logging.getLogger(__name__)

def visualize_detailed_morphology(
    image_path_or_array: Any,
    main_props_s3: Optional[RegionProperties], 
    egg_fits_s4: Optional[List[Optional[Dict[str, Any]]]],
    raw_body_mask_s4: Optional[np.ndarray],
    filled_body_mask_s4: Optional[np.ndarray], 
    ideal_body_mask_s4: Optional[np.ndarray], 
    output_visualization_path: str
):
    try:
        if isinstance(image_path_or_array, str):
            image_orig = skimage.io.imread(image_path_or_array)
        elif isinstance(image_path_or_array, np.ndarray):
            image_orig = image_path_or_array.copy() 
        else:
            logger.error("Invalid image_path_or_array type for S4 visualization.")
            return

        image_float = skimage.util.img_as_float(image_orig)
        if image_float.ndim == 2: 
            image_rgb = skimage.color.gray2rgb(image_float)
        elif image_float.ndim == 3 and image_float.shape[2] == 4: 
            image_rgb = skimage.color.rgba2rgb(image_float)
        elif image_float.ndim == 3 and image_float.shape[2] == 3: 
            image_rgb = image_float
        else: 
            logger.warning(f"Unexpected image format for S4 visualization. Shape: {image_float.shape}. Trying first channel.")
            if image_float.ndim == 3 and image_float.shape[2] > 0:
                 image_rgb = skimage.color.gray2rgb(image_float[:,:,0])
            else: 
                 logger.error(f"Cannot convert image of shape {image_float.shape} to RGB for S4 visualization.")
                 return

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image_rgb, cmap='gray')
        ax.set_axis_off()

        if main_props_s3 and \
           hasattr(main_props_s3, 'centroid') and \
           hasattr(main_props_s3, 'orientation') and \
           hasattr(main_props_s3, 'major_axis_length') and \
           main_props_s3.major_axis_length > 0 and \
           hasattr(main_props_s3, '_axis_vector_pointing_posterior'): # Check for the attached vector

            y0_row, x0_col = main_props_s3.centroid 
            # Orientation from props is fine for drawing the axis line itself
            orientation_rad_for_line = main_props_s3.orientation 
            major_axis_len_val = main_props_s3.major_axis_length
            major_axis_half = major_axis_len_val / 2.0
            
            # Draw the major axis line
            axis_vector_for_line = np.array([np.cos(orientation_rad_for_line), np.sin(orientation_rad_for_line)])
            p1_major_axis_r = y0_row + major_axis_half * axis_vector_for_line[0]
            p1_major_axis_c = x0_col + major_axis_half * axis_vector_for_line[1]
            p2_major_axis_r = y0_row - major_axis_half * axis_vector_for_line[0]
            p2_major_axis_c = x0_col - major_axis_half * axis_vector_for_line[1]
            
            ax.plot((p1_major_axis_c, p2_major_axis_c), (p1_major_axis_r, p2_major_axis_r), 
                    '--y', linewidth=1.2, label="Major Axis") # Removed (S3) for clarity

            # Use the _axis_vector_pointing_posterior for the posterior end marker
            # This vector is already correctly oriented.
            _axis_vec_posterior = main_props_s3._axis_vector_pointing_posterior 
            post_end_r = y0_row + major_axis_half * _axis_vec_posterior[0]
            post_end_c = x0_col + major_axis_half * _axis_vec_posterior[1]
            
            ax.plot(post_end_c, post_end_r, 'oy', markersize=5, label="Posterior End")
        elif main_props_s3:
            logger.warning("main_props_s3 missing attributes for posterior end visualization (e.g., _axis_vector_pointing_posterior).")


        if egg_fits_s4:
            for i, egg_data in enumerate(egg_fits_s4):
                if egg_data and 'center_x' in egg_data and 'radius' in egg_data and egg_data['radius'] > 0:
                    egg_circle = patches.Circle(
                        (egg_data['center_x'], egg_data['center_y']), 
                        egg_data['radius'],
                        fill=False, edgecolor='magenta', linewidth=1.5, 
                        label=f"Fitted Egg {i+1}" if i == 0 else None 
                    )
                    ax.add_patch(egg_circle)

        if raw_body_mask_s4 is not None and raw_body_mask_s4.sum() > 0:
            contours_raw = skimage.measure.find_contours(raw_body_mask_s4.astype(np.uint8), 0.5) 
            for n, contour in enumerate(contours_raw):
                ax.plot(contour[:, 1], contour[:, 0], color='cyan', linestyle=':', linewidth=1.0, 
                        label="Raw Body Contour" if n == 0 else None) # Removed (S4)
        
        contour_to_plot_label = "Ideal Body Contour" # Removed (S4)
        mask_for_final_body_contour = ideal_body_mask_s4
        final_body_contour_color = 'blue'
        
        if ideal_body_mask_s4 is None or ideal_body_mask_s4.sum() == 0:
            logger.debug("Ideal body mask not available or empty for visualization, falling back to filled body mask.")
            mask_for_final_body_contour = filled_body_mask_s4
            contour_to_plot_label = "Filled Body Contour (Fallback)" # Removed (S4)
            final_body_contour_color = 'lime' 

        if mask_for_final_body_contour is not None and mask_for_final_body_contour.sum() > 0:
            contours_final_body = skimage.measure.find_contours(mask_for_final_body_contour.astype(np.uint8), 0.5)
            for n, contour in enumerate(contours_final_body):
                ax.plot(contour[:, 1], contour[:, 0], color=final_body_contour_color, linestyle='-', linewidth=1.5, 
                        label=contour_to_plot_label if n == 0 else None)
        else:
            logger.warning("Neither ideal nor filled body mask available for final contour visualization.")

        title_str = os.path.basename(image_path_or_array) if isinstance(image_path_or_array, str) else "Detailed Morphology"
        ax.set_title(title_str, fontsize=9)
        
        handles, labels = ax.get_legend_handles_labels()
        if handles: 
            unique_labels_dict = {}
            unique_handles_list = []
            for handle, label_text in zip(handles, labels):
                if label_text not in unique_labels_dict:
                    unique_labels_dict[label_text] = handle
                    unique_handles_list.append(handle)
            ax.legend(unique_handles_list, unique_labels_dict.keys(), fontsize='x-small', loc='upper right')

        plt.tight_layout(pad=0.5)
        os.makedirs(os.path.dirname(output_visualization_path), exist_ok=True)
        plt.savefig(output_visualization_path, dpi=150, bbox_inches='tight')
        logger.debug(f"S4 Visualization saved to: {output_visualization_path}")

    except FileNotFoundError:
        logger.error(f"Error visualizing S4: Image not found at '{image_path_or_array}'")
    except Exception as e:
        img_ref = image_path_or_array if isinstance(image_path_or_array, str) else 'image array'
        logger.exception(f"Error during S4 visualization of '{img_ref}': {e}")
    finally:
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)

if __name__ == '__main__':
    import skimage 
    logging.basicConfig(level=logging.DEBUG)
    test_h, test_w = 200, 250
    test_image_s4_vis = np.zeros((test_h, test_w), dtype=np.uint8)
    rr_body, cc_body = skimage.draw.ellipse(100, 125, 70, 40, rotation=np.deg2rad(15))
    test_image_s4_vis[rr_body, cc_body] = 150
    rr_hole, cc_hole = skimage.draw.ellipse(90, 110, 15, 10, rotation=np.deg2rad(15))
    test_image_s4_vis[rr_hole, cc_hole] = 0
    rr_egg_zone, cc_egg_zone = skimage.draw.ellipse(100, 50, 30, 25, rotation=np.deg2rad(0))
    test_image_s4_vis[rr_egg_zone, cc_egg_zone] = 200

    class DummyPropsS4Vis(RegionProperties): 
        def __init__(self, centroid, orientation, major_axis_length, axis_vector_posterior):
            self.centroid = centroid
            self.orientation = orientation
            self.major_axis_length = major_axis_length
            self._axis_vector_pointing_posterior = axis_vector_posterior # Attach the vector
            # Super init for RegionProperties
            # Create a minimal slice and intensity_image for super().__init__
            dummy_slice = tuple(slice(0,1) for _ in centroid) 
            dummy_intensity_image = np.array([[0]]) 
            super().__init__(slice=dummy_slice, label=1, intensity_image=dummy_intensity_image, cache_active=False)


    # Simulate a case where positive projection side is posterior
    axis_vec_post = np.array([np.cos(np.deg2rad(15)), np.sin(np.deg2rad(15))])
    main_p_s3_vis = DummyPropsS4Vis(
        centroid=(100, 125), 
        orientation=np.deg2rad(15), 
        major_axis_length=140,
        axis_vector_posterior=axis_vec_post
    )
    egg_fits_vis = [{'center_x': 40, 'center_y': 90, 'radius': 18, 'area': np.pi*18**2}, {'center_x': 65, 'center_y': 110, 'radius': 15, 'area': np.pi*15**2}]
    raw_m_vis = np.zeros_like(test_image_s4_vis); raw_m_vis[rr_body, cc_body] = 255; raw_m_vis[rr_hole, cc_hole] = 0
    filled_m_vis = np.zeros_like(test_image_s4_vis); filled_m_vis[rr_body, cc_body] = 255
    
    ideal_m_test_vis = np.zeros_like(test_image_s4_vis)
    rr_ideal_vis, cc_ideal_vis = skimage.draw.ellipse(100,110, 75, 45, rotation=np.deg2rad(10)) 
    ideal_m_test_vis[rr_ideal_vis, cc_ideal_vis] = 255

    output_vis_path_main = "temp_stage4_visualization_test_ideal_posterior_fix.png"
    visualize_detailed_morphology(
        test_image_s4_vis, main_p_s3_vis, egg_fits_vis, raw_m_vis, filled_m_vis, ideal_m_test_vis, output_vis_path_main
    )
    if os.path.exists(output_vis_path_main): logger.info(f"Stage 4 test visualization (posterior fix) saved to {output_vis_path_main}")
    else: logger.error("Stage 4 test visualization (posterior fix) FAILED to save.")