# src/core/feature_calculators/__init__.py
#
# Public API of the feature_calculators subpackage.
# Import from the sub-module directly for explicit dependency tracking:
#
#   from src.core.feature_calculators.morphology_calculator import calculate_detailed_body_morphology
#   from src.core.feature_calculators.quality_calculator import calculate_quality_features
#   from src.core.feature_calculators.shape_descriptors import calculate_shape_descriptors
#   from src.core.feature_calculators.bio_features_calculator import extract_ml_features
#
# Sub-module public API
# ----------------------
# morphology_calculator   : calculate_detailed_body_morphology
# quality_calculator      : calculate_quality_features
# shape_descriptors       : calculate_shape_descriptors
# bio_features_calculator : extract_ml_features, calculate_posterior_attributes,
#                           DEFAULT_FEATURE_NAMES
