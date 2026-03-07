# src/visualization/feature_visualization.py
import skimage.io
import skimage.color
import skimage.util
import skimage.measure # For find_contours
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def visualize_ml_features(
    image_path_or_array: Any, 
    main_props: Optional[Any], 
    egg_area_props: Optional[Any], 
    output_visualization_path: str,
    ml_feature_params: Optional[Dict[str, Any]] = None # Used for egg_posterior_pixel_fraction
):
    """
    Visualizes an image with overlays for the main detected body and potential egg area.
    Highlights the main body contour, its major axis, and the posterior end.
    Overlays the contour for the detected egg area if present.
    Draws the contour of the posterior pixel area of the main body based on configuration.
    """
    try:
        if isinstance(image_path_or_array, str):
            image_orig = skimage.io.imread(image_path_or_array)
        elif isinstance(image_path_or_array, np.ndarray):
            image_orig = image_path_or_array
        else:
            logger.error("Invalid image_path_or_array type for visualization. Must be str or np.ndarray.")
            return

        img_shape_for_draw = image_orig.shape[:2] 

        image_float = skimage.util.img_as_float(image_orig)
        if image_float.ndim == 2: 
            image_rgb = skimage.color.gray2rgb(image_float)
        elif image_float.ndim == 3 and image_float.shape[2] == 4: 
            image_rgb = skimage.color.rgba2rgb(image_float)
        elif image_float.ndim == 3 and image_float.shape[2] == 3: 
            image_rgb = image_float
        else: 
            logger.warning(f"Unexpected image format for visualization. Shape: {image_float.shape}. Trying first channel.")
            if image_float.ndim == 3 and image_float.shape[2] > 0:
                 image_rgb = skimage.color.gray2rgb(image_float[:,:,0])
            else: 
                 logger.error(f"Cannot convert image of shape {image_float.shape} to RGB for visualization.")
                 return
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_rgb, cmap='gray') 
        ax.set_axis_off()
        
        if main_props and hasattr(main_props, 'coords'):
            ax.plot(main_props.coords[:, 1], main_props.coords[:, 0], 
                    linestyle='none', marker='.', markersize=1, color='cyan', alpha=0.5, label="Main Body")

        if egg_area_props and hasattr(egg_area_props, 'coords'): 
            ax.plot(egg_area_props.coords[:, 1], egg_area_props.coords[:, 0], 
                    linestyle='none', marker='.', markersize=1.2, color='magenta', alpha=0.6, label="Egg Area")

        if main_props and \
           hasattr(main_props, 'centroid') and \
           hasattr(main_props, 'orientation') and \
           hasattr(main_props, 'major_axis_length') and \
           hasattr(main_props, 'coords') and \
           main_props.major_axis_length > 0:
            
            y0_row, x0_col = main_props.centroid 
            orientation_rad = main_props.orientation 
            major_axis_len_val = main_props.major_axis_length
            major_axis_half = major_axis_len_val / 2.0

            axis_vector_rc_unit = np.array([np.cos(orientation_rad), np.sin(orientation_rad)]) 

            p1_major_axis_r = y0_row + major_axis_half * axis_vector_rc_unit[0]
            p1_major_axis_c = x0_col + major_axis_half * axis_vector_rc_unit[1]
            p2_major_axis_r = y0_row - major_axis_half * axis_vector_rc_unit[0]
            p2_major_axis_c = x0_col - major_axis_half * axis_vector_rc_unit[1]
            
            ax.plot((p1_major_axis_c, p2_major_axis_c), (p1_major_axis_r, p2_major_axis_r), 
                    '--y', linewidth=1.2, label="Major Axis")

            posterior_area_ratio_for_viz = 0.5 
            if ml_feature_params and main_props.area > 0: 
                coords = main_props.coords 
                centroid_arr = np.array(main_props.centroid) 
                relative_coords = coords - centroid_arr 
                projections = np.dot(relative_coords, axis_vector_rc_unit) 
                posterior_pixels_count = np.sum(projections > 0) 
                if main_props.area > 0:
                    posterior_area_ratio_for_viz = posterior_pixels_count / main_props.area
            
            positive_projection_side_is_posterior = (posterior_area_ratio_for_viz >= 0.5)
            
            post_end_r, post_end_c = (p1_major_axis_r, p1_major_axis_c) if positive_projection_side_is_posterior else (p2_major_axis_r, p2_major_axis_c)
            ax.plot(post_end_c, post_end_r, 'oy', markersize=5, label="Posterior End")

            _axis_vector_pointing_posterior_viz = axis_vector_rc_unit.copy()
            if not positive_projection_side_is_posterior:
                _axis_vector_pointing_posterior_viz = -_axis_vector_pointing_posterior_viz

            if main_props.area > 0:
                # Get posterior fraction from ml_feature_params (config)
                # Default to 0.45 if not specified in params for visualization
                egg_posterior_fraction_viz = 0.45 
                if ml_feature_params and 'egg_posterior_pixel_fraction' in ml_feature_params:
                    egg_posterior_fraction_viz = ml_feature_params.get('egg_posterior_pixel_fraction', 0.45)
                
                percentile_to_cut_viz = (1.0 - egg_posterior_fraction_viz) * 100.0

                # Check if _posterior_coords_for_egg_search was attached (using the config fraction)
                # For visualization, it's safer to recalculate using the viz-specific fraction if needed,
                # or ensure ml_feature_params given to visualize_ml_features is the exact one used for extraction.
                # Here, we recalculate to ensure visualization uses egg_posterior_fraction_viz.
                
                coords_main_body_viz = main_props.coords
                centroid_main_body_viz = np.array(main_props.centroid)
                relative_coords_for_slice_viz = coords_main_body_viz - centroid_main_body_viz
                projections_for_slice_viz = np.dot(relative_coords_for_slice_viz, _axis_vector_pointing_posterior_viz)
                
                posterior_coords_viz = np.array([]) # Initialize
                if len(projections_for_slice_viz) > 0:
                    projection_threshold_val_viz = np.percentile(projections_for_slice_viz, percentile_to_cut_viz)
                    posterior_indices_viz = projections_for_slice_viz > projection_threshold_val_viz
                    posterior_coords_viz = coords_main_body_viz[posterior_indices_viz]
                
                if posterior_coords_viz.shape[0] > 0:
                    temp_posterior_pixel_mask = np.zeros(img_shape_for_draw, dtype=np.uint8)
                    valid_rows = np.clip(posterior_coords_viz[:, 0].astype(int), 0, img_shape_for_draw[0] - 1)
                    valid_cols = np.clip(posterior_coords_viz[:, 1].astype(int), 0, img_shape_for_draw[1] - 1)
                    temp_posterior_pixel_mask[valid_rows, valid_cols] = 255
                        
                    contours_posterior_area = skimage.measure.find_contours(temp_posterior_pixel_mask, 0.5) 
                        
                    for n, contour in enumerate(contours_posterior_area):
                        label = f"Posterior {egg_posterior_fraction_viz*100:.0f}% Area" if n == 0 else None 
                        ax.plot(contour[:, 1], contour[:, 0], color='lime', linestyle=':', linewidth=1.5, label=label)
            
        title_str = os.path.basename(image_path_or_array) if isinstance(image_path_or_array, str) else "Processed Image"
        ax.set_title(title_str, fontsize=9)
        
        handles, labels = ax.get_legend_handles_labels()
        if handles: 
            unique_labels = {}
            unique_handles = []
            for handle, label_text in zip(handles, labels):
                if label_text not in unique_labels:
                    unique_labels[label_text] = handle
                    unique_handles.append(handle)
            ax.legend(unique_handles, unique_labels.keys(), fontsize='x-small', loc='upper right')

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
