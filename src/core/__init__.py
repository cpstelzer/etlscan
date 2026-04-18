# src/core/__init__.py
#
# Each sub-module in src/core/ is self-contained.  Import from the
# sub-module directly to keep dependency chains explicit:
#
#   from src.core.image_io_operations import load_image
#   from src.core.sharpness_assessment import calculate_sharpness_metrics
#
# Public API surfaces per sub-module
# -----------------------------------
# image_io_operations   : load_image, save_image, convert_to_grayscale,
#                         load_flat_field_assets, apply_flat_field_correction,
#                         FlatFieldCorrectionAssets
# image_enhancement     : subtract_images, apply_threshold, apply_blur,
#                         apply_morphological_operation
# masking_operations    : create_circular_mask, apply_mask, mask_well_area,
#                         detect_black_mask_artifact
# well_analysis         : create_well_mask_black_plate,
#                         calculate_well_brightness_peak, match_well_brightness
# sharpness_assessment  : calculate_sharpness_metrics,
#                         normalize_and_combine_sharpness,
#                         AVAILABLE_SHARPNESS_METRICS
# region_props_calculator: find_contours, calculate_contour_measurements,
#                          filter_contours_by_area
# segmentation          : generate_binary_mask_from_float_image,
#                         segment_main_body_from_image_float
# particle_operations   : crop_particle_from_arrays, compute_adaptive_padding,
#                         normalized_crop_config, detect_cropped_particle_on_boundary
