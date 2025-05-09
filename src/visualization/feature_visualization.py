# src/visualization/feature_visualization.py
import skimage.io
import skimage.color
import skimage.util
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any # Any for skimage.measure._regionprops.RegionProperties
import logging

logger = logging.getLogger(__name__)

def visualize_ml_features(
    image_path_or_array: Any, # Can be path (str) or loaded image array (np.ndarray)
    main_props: Optional[Any], # skimage.measure._regionprops.RegionProperties object
    egg_pair_props: Optional[Any], # skimage.measure._regionprops.RegionProperties object
    output_visualization_path: str,
    ml_feature_params: Optional[Dict[str, Any]] = None # For posterior_area_ratio re-calc if needed
):
    """
    Visualizes an image with overlays for the main detected body and potential egg pair.
    Highlights the main body contour, its major axis, and the posterior end.
    Overlays the contour for the detected egg pair region if present.
    This version aligns axis drawing and posterior logic with skimage.measure.orientation
    (angle from y-axis/row-axis, CCW).
    """
    try:
        if isinstance(image_path_or_array, str):
            image_orig = skimage.io.imread(image_path_or_array)
        elif isinstance(image_path_or_array, np.ndarray):
            image_orig = image_path_or_array
        else:
            logger.error("Invalid image_path_or_array type for visualization. Must be str or np.ndarray.")
            return

        image_float = skimage.util.img_as_float(image_orig)
        if image_float.ndim == 2: 
            image_rgb = skimage.color.gray2rgb(image_float)
        elif image_float.ndim == 3 and image_float.shape[2] == 4: 
            image_rgb = skimage.color.rgba2rgb(image_float)
        elif image_float.ndim == 3 and image_float.shape[2] == 3: 
            image_rgb = image_float
        else:
            logger.warning(f"Unexpected image format for visualization. Shape: {image_float.shape}")
            if image_float.ndim == 3 and image_float.shape[2] > 1:
                 image_rgb = skimage.color.gray2rgb(image_float[:,:,0])
            else: 
                 logger.error("Cannot convert image to RGB for visualization.")
                 return
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_rgb, cmap='gray') 
        ax.set_axis_off()
        
        if main_props and hasattr(main_props, 'coords'):
            ax.plot(main_props.coords[:, 1], main_props.coords[:, 0], 
                    linestyle='none', marker='.', markersize=1, color='cyan', alpha=0.5, label="Main Body")

        if egg_pair_props and hasattr(egg_pair_props, 'coords'):
            ax.plot(egg_pair_props.coords[:, 1], egg_pair_props.coords[:, 0], 
                    linestyle='none', marker='.', markersize=1.2, color='magenta', alpha=0.6, label="Egg Pair")

        if main_props and \
           hasattr(main_props, 'centroid') and \
           hasattr(main_props, 'orientation') and \
           hasattr(main_props, 'major_axis_length') and \
           hasattr(main_props, 'coords') and \
           main_props.major_axis_length > 0:
            
            y0_row, x0_col = main_props.centroid # (row, col)
            # orientation_rad is angle from y-axis (row-axis), CCW, from skimage
            orientation_rad = main_props.orientation 
            major_axis_half = main_props.major_axis_length / 2.0

            # Consistent with original ws_identify_ideal_fems and skimage orientation:
            # delta_row = L * cos(orientation_rad)
            # delta_col = L * sin(orientation_rad)
            d_row = major_axis_half * np.cos(orientation_rad)
            d_col = major_axis_half * np.sin(orientation_rad)
            
            # Endpoints in (row, col) image coordinates
            y1_row, x1_col = y0_row + d_row, x0_col + d_col
            y2_row, x2_col = y0_row - d_row, x0_col - d_col
            
            # ax.plot expects (x_coordinates, y_coordinates)
            ax.plot((x1_col, x2_col), (y1_row, y2_row), '--y', linewidth=1.2, label="Major Axis")

            # Determine posterior end for marker
            # The axis_vector_rc should be a unit vector (d_row_unit, d_col_unit)
            # consistent with the (d_row, d_col) calculated above.
            axis_vector_rc = np.array([np.cos(orientation_rad), np.sin(orientation_rad)]) # (d_row_unit, d_col_unit)

            posterior_area_ratio_for_viz = 0.5 # Default assumption
            # Re-calculate posterior_area_ratio for visualization for consistency with feature extraction
            if ml_feature_params and main_props.area > 0: 
                coords = main_props.coords # Nx2 array (row, col)
                centroid_arr = np.array(main_props.centroid) # (row, col)
                relative_coords = coords - centroid_arr # Each row is (rel_row, rel_col)
                
                # Project relative_coords onto axis_vector_rc
                projections = np.dot(relative_coords, axis_vector_rc) 
                
                posterior_pixels_count = np.sum(projections > 0) # Pixels on the "positive projection" side
                if main_props.area > 0:
                    posterior_area_ratio_for_viz = posterior_pixels_count / main_props.area
            
            positive_projection_side_is_posterior = (posterior_area_ratio_for_viz >= 0.5)
            
            # Determine which end of the drawn axis (y1_row,x1_col) or (y2_row,y2_col) is posterior
            # (y1_row, x1_col) corresponds to +d_row, +d_col along axis_vector_rc
            # (y2_row, x2_col) corresponds to -d_row, -d_col along axis_vector_rc
            
            if positive_projection_side_is_posterior:
                post_x_marker, post_y_marker = x1_col, y1_row
            else:
                post_x_marker, post_y_marker = x2_col, y2_row
            
            ax.plot(post_x_marker, post_y_marker, 'oy', markersize=5, label="Posterior End")

        title_str = os.path.basename(image_path_or_array) if isinstance(image_path_or_array, str) else "Processed Image"
        ax.set_title(title_str, fontsize=9)
        
        if ax.get_legend_handles_labels()[0]: 
            ax.legend(fontsize='x-small', loc='upper right')

        plt.tight_layout(pad=0.5)
        
        os.makedirs(os.path.dirname(output_visualization_path), exist_ok=True)
        
        plt.savefig(output_visualization_path, dpi=150, bbox_inches='tight')
        logger.debug(f"Visualization saved to: {output_visualization_path}")

    except FileNotFoundError:
        logger.error(f"Error visualizing: Image not found at '{image_path_or_array}'")
    except Exception as e:
        logger.exception(f"Error during visualization of '{image_path_or_array if isinstance(image_path_or_array, str) else 'image array'}': {e}")
    finally:
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)


if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG)

    from src.core.ml_feature_extractor import extract_ml_features 
    from src.utils.config_loader import get_config 

    try:
        cfg = get_config() 
        ml_params = cfg.get('stage_03_ml_classification', {}).get('feature_extraction_params', {})
        if not ml_params: 
            ml_params = {
                'gaussian_sigma': 1.0, 'otsu_threshold_multiplier': 0.5, 'closing_selem_size': 3,
                'min_noise_size': 50, 'min_significant_area': 100, 
                'egg_threshold_quantile': 0.90, 'egg_opening_selem_size': 1,
                'egg_pair_min_area': 50, 'egg_pair_max_area': 5000,
                'egg_pair_min_solidity': 0.60, 'egg_pair_max_solidity': 0.99,
                'egg_pair_min_eccentricity': 0.2, 'max_egg_distance_factor': 1.2,
                'posterior_projection_threshold': -0.2, 'orientation_diff_tolerance_deg': 60
            }
    except Exception as e:
        print(f"Could not load config for test: {e}. Using hardcoded params.")
        ml_params = {
                'gaussian_sigma': 1.0, 'otsu_threshold_multiplier': 0.5, 'closing_selem_size': 3,
                'min_noise_size': 50, 'min_significant_area': 100, 
                'egg_threshold_quantile': 0.90, 'egg_opening_selem_size': 1,
                'egg_pair_min_area': 50, 'egg_pair_max_area': 5000,
                'egg_pair_min_solidity': 0.60, 'egg_pair_max_solidity': 0.99,
                'egg_pair_min_eccentricity': 0.2, 'max_egg_distance_factor': 1.2,
                'posterior_projection_threshold': -0.2, 'orientation_diff_tolerance_deg': 60
            }

    test_vis_image = np.zeros((200, 200), dtype=np.uint8)
    # Body rotated 0 degrees from Y-axis (vertical body)
    rr_body_vert, cc_body_vert = skimage.draw.ellipse(100, 100, 40, 20, rotation=np.deg2rad(0)) 
    test_vis_image_vert = test_vis_image.copy()
    test_vis_image_vert[rr_body_vert, cc_body_vert] = 150
    
    # Body rotated 90 degrees from Y-axis (horizontal body)
    rr_body_horiz, cc_body_horiz = skimage.draw.ellipse(100, 100, 40, 20, rotation=np.deg2rad(90))
    test_vis_image_horiz = test_vis_image.copy()
    test_vis_image_horiz[rr_body_horiz, cc_body_horiz] = 150

    temp_image_file_vert = "temp_vis_test_image_vert.png"
    skimage.io.imsave(temp_image_file_vert, skimage.util.img_as_ubyte(test_vis_image_vert))
    temp_image_file_horiz = "temp_vis_test_image_horiz.png"
    skimage.io.imsave(temp_image_file_horiz, skimage.util.img_as_ubyte(test_vis_image_horiz))

    # Test vertical body (orientation should be ~0)
    features_v, main_p_v, egg_p_v = extract_ml_features(
        temp_image_file_vert, ml_params, image_array_gray=skimage.util.img_as_float(test_vis_image_vert)
    )
    if main_p_v: logger.info(f"Vertical body orientation: {np.rad2deg(main_p_v.orientation):.2f} degrees")
    output_vis_file_v = "temp_output_visualization_vert.png"
    visualize_ml_features(temp_image_file_vert, main_p_v, egg_p_v, output_vis_file_v, ml_feature_params=ml_params)
    if os.path.exists(output_vis_file_v): logger.info(f"Vertical test visualization saved to {output_vis_file_v}")

    # Test horizontal body (orientation should be ~pi/2 or ~-pi/2)
    features_h, main_p_h, egg_p_h = extract_ml_features(
        temp_image_file_horiz, ml_params, image_array_gray=skimage.util.img_as_float(test_vis_image_horiz)
    )
    if main_p_h: logger.info(f"Horizontal body orientation: {np.rad2deg(main_p_h.orientation):.2f} degrees")
    output_vis_file_h = "temp_output_visualization_horiz.png"
    visualize_ml_features(temp_image_file_horiz, main_p_h, egg_p_h, output_vis_file_h, ml_feature_params=ml_params)
    if os.path.exists(output_vis_file_h): logger.info(f"Horizontal test visualization saved to {output_vis_file_h}")

    if os.path.exists(temp_image_file_vert): os.remove(temp_image_file_vert)
    if os.path.exists(temp_image_file_horiz): os.remove(temp_image_file_horiz)