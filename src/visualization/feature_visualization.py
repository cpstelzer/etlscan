# src/visualization/feature_visualization.py
import skimage.io
import skimage.color
import skimage.util
import skimage.measure # For find_contours
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Dict, Any, Sequence
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


def plot_probability_histogram(
    probabilities: Sequence[float],
    output_path: str,
    *,
    labels: Optional[Sequence[str]] = None,
    threshold: Optional[float] = None,
    bins: int = 20,
) -> None:
    """Render a histogram of ML probabilities for a run."""
    try:
        if not probabilities:
            logger.warning("No probabilities provided for histogram plot at %s", output_path)
            return

        probabilities_arr = np.asarray(probabilities, dtype=float)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(probabilities_arr, bins=bins, color='steelblue', alpha=0.8)
        ax.set_xlabel('Positive-class probability')
        ax.set_ylabel('Particle count')
        ax.set_title('Stage 3 classifier probability distribution')

        if threshold is not None:
            ax.axvline(threshold, color='crimson', linestyle='--', label=f'Threshold = {threshold:.2f}')

        if labels is not None:
            try:
                label_counts = {}
                for lbl in labels:
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
                annotation = ", ".join(f"{k}: {v}" for k, v in label_counts.items())
                ax.text(0.02, 0.95, f"Counts: {annotation}", transform=ax.transAxes,
                        va='top', ha='left', fontsize='x-small')
            except Exception as exc:
                logger.debug("Failed to annotate label counts on histogram: %s", exc)

        if threshold is not None:
            ax.legend(loc='upper right', fontsize='small')

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        logger.info("Probability histogram saved to %s", output_path)
    except Exception as exc:
        logger.exception("Failed to plot probability histogram: %s", exc)
    finally:
        if 'fig' in locals():
            plt.close(fig)


def plot_roc_curve(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    output_path: str,
    *,
    positive_label: int = 1,
) -> None:
    """Plot a ROC-style curve for the classifier on a held-out set."""
    try:
        from sklearn.metrics import roc_curve, auc

        if not y_true or not y_scores:
            logger.warning("Insufficient data for ROC plot at %s", output_path)
            return

        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=positive_label)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Stage 3 classifier ROC summary')
        ax.legend(loc='lower right')
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        logger.info("ROC plot saved to %s", output_path)
    except Exception as exc:
        logger.exception("Failed to plot ROC curve: %s", exc)
    finally:
        if 'fig' in locals():
            plt.close(fig)


if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG)

    from src.core.ml_feature_extractor import extract_ml_features 
    from src.utils.config_loader import get_config 

    try:
        cfg = get_config() 
        # ml_feature_params will now be read by visualize_ml_features for egg_posterior_pixel_fraction
        ml_params_for_extraction_and_viz = cfg.get('stage_03_ml_classification', {}).get('feature_extraction_params', {})
        if not ml_params_for_extraction_and_viz or 'egg_posterior_pixel_fraction' not in ml_params_for_extraction_and_viz: 
            logger.warning("ML feature extraction parameters (or egg_posterior_pixel_fraction) "
                           "not found in config for test. Using hardcoded test params.")
            ml_params_for_extraction_and_viz = {
                'gaussian_sigma': 1.0, 'otsu_threshold_multiplier': 0.5, 'closing_selem_size': 3,
                'min_noise_size': 50, 'min_significant_area': 100, 
                'egg_posterior_pixel_fraction': 0.45, # Example for testing
                'egg_threshold_quantile': 0.90, 
                'egg_min_area': 50, 'egg_max_area': 5000,
                'egg_min_solidity': 0.60, 'egg_max_solidity': 0.99,
                'egg_min_eccentricity': 0.2, 
                'orientation_diff_tolerance_deg': 60
            }
    except Exception as e:
        print(f"Could not load config for test: {e}. Using hardcoded params.")
        ml_params_for_extraction_and_viz = { 
            'gaussian_sigma': 1.0, 'otsu_threshold_multiplier': 0.5, 'closing_selem_size': 3,
            'min_noise_size': 50, 'min_significant_area': 100, 
            'egg_posterior_pixel_fraction': 0.45, # Example for testing
            'egg_threshold_quantile': 0.90,
            'egg_min_area': 50, 'egg_max_area': 5000,
            'egg_min_solidity': 0.60, 'egg_max_solidity': 0.99,
            'egg_min_eccentricity': 0.2,
            'orientation_diff_tolerance_deg': 60
        }

    test_image_v = np.zeros((200, 200), dtype=np.uint8)
    rr_body_v, cc_body_v = skimage.draw.ellipse(100, 100, 70, 25, rotation=np.deg2rad(10)) 
    test_image_v[rr_body_v, cc_body_v] = 150
    rr_heavy, cc_heavy = skimage.draw.ellipse(100 + int(70*0.6*np.cos(np.deg2rad(10))), 
                                              100 + int(70*0.6*np.sin(np.deg2rad(10))), 
                                              15,10, rotation=np.deg2rad(10))
    test_image_v[rr_heavy, cc_heavy] = 160
    
    rr_egg_v, cc_egg_v = skimage.draw.ellipse(100 + int(70*0.5*np.cos(np.deg2rad(10))) , 
                                              100 + int(70*0.5*np.sin(np.deg2rad(10))), 
                                              15, 20, rotation=np.deg2rad(10+90))
    test_image_v[rr_egg_v, cc_egg_v] = 220 

    temp_image_file_v = "temp_vis_test_image_pixel_slice_contour_config.png"
    skimage.io.imsave(temp_image_file_v, skimage.util.img_as_ubyte(test_image_v))
    
    # Pass the same ml_params to extraction and visualization
    features_v, main_p_v, egg_area_p_v = extract_ml_features(
        temp_image_file_v, ml_params_for_extraction_and_viz, 
        image_array_gray=skimage.util.img_as_float(test_image_v)
    )
    if main_p_v: logger.info(f"Test body orientation: {np.rad2deg(main_p_v.orientation):.2f} degrees")
    output_vis_file_v = "temp_output_visualization_pixel_slice_contour_config.png"
    visualize_ml_features(temp_image_file_v, main_p_v, egg_area_p_v, output_vis_file_v, 
                          ml_feature_params=ml_params_for_extraction_and_viz) # Pass params here
    if os.path.exists(output_vis_file_v): 
        logger.info(f"Test visualization with pixel slice (config fraction) contour saved to {output_vis_file_v}")

    problem_image_filename = "problem_image_for_viz_test.png" 
    logger.info(f"If '{problem_image_filename}' (your example image) exists in the current dir, it will be processed.")
    if os.path.exists(problem_image_filename):
        problem_img_array = skimage.io.imread(problem_image_filename, as_gray=True)
        problem_img_array_float = skimage.util.img_as_float(problem_img_array)

        logger.info(f"Processing your provided image: {problem_image_filename}")
        features_prob, main_p_prob, egg_area_p_prob = extract_ml_features(
            problem_image_filename, ml_params_for_extraction_and_viz, 
            image_array_gray=problem_img_array_float
        )
        if features_prob and main_p_prob:
            logger.info(f"Problem image - Main Orientation (deg): {np.rad2deg(main_p_prob.orientation)}")
            logger.info(f"Problem image - Posterior Area Ratio: {features_prob['posterior_area_ratio']}")
            logger.info(f"Problem image - Total Egg Area: {features_prob['total_egg_area']}")
            if hasattr(main_p_prob, '_posterior_coords_for_egg_search'): 
                 num_posterior_pixels = main_p_prob._posterior_coords_for_egg_search.shape[0]
                 expected_frac_prob = ml_params_for_extraction_and_viz.get('egg_posterior_pixel_fraction',0.45)
                 logger.info(f"Problem image - Num posterior {expected_frac_prob*100:.0f}% pixels: {num_posterior_pixels}")

            output_vis_file_prob = "output_visualization_problem_image_pixel_slice_config.png"
            visualize_ml_features(
                problem_image_filename, 
                main_p_prob, 
                egg_area_p_prob, 
                output_vis_file_prob, 
                ml_feature_params=ml_params_for_extraction_and_viz # Pass params here
            )
            if os.path.exists(output_vis_file_prob): 
                logger.info(f"Visualization for problem image (pixel slice config frac) saved to {output_vis_file_prob}")
        else:
            logger.error(f"Could not extract features from {problem_image_filename} for visualization test.")

    if os.path.exists(temp_image_file_v): os.remove(temp_image_file_v)