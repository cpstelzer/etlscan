# `focussed_and_filtered.csv` schema

The Stage 2 \u2192 Stage 3 hand-off is the last point at which particle features are
enriched before machine learning. The table below enumerates every numeric column
that is persisted in `focussed_and_filtered.csv`, grouped by the stage/component
that generates it.

## Metadata columns tagged for ML filtering
These identifier fields are renamed with the `meta__` prefix inside Stage 3 so
that automated preprocessing can ignore them safely.

- `meta__SAMPLE_ID`
- `meta__source_image_filename`
- `meta__crop_filename`
- `meta__final_crop_filename`
- `meta__track_id`

## Stage 1 detection & tracking outputs
- `frame`
- `X`, `Y`
- `x_mm`, `y_mm`, `z_mm`
- `dioptries`
- `local_threshold_value`
- `bounding_x`, `bounding_y`, `bounding_w`, `bounding_h`
- `rot_rect_width`, `rot_rect_height`, `rot_rect_angle`, `rot_rect_area`
- `s1_area`, `s1_perimeter`, `s1_orientation`, `s1_aspect_ratio`
- `s1_solidity`, `s1_eccentricity`, `s1_feret_diameter`, `s1_circularity`
- `s1_equivalent_diameter`

## Stage 2 morphology features (renamed from `_s4`)
- `main_area_raw`
- `solidity`
- `compactness_rect`
- `extent_raw`
- `eccentricity_raw`
- `equivalent_diameter_raw`
- `major_axis_length_raw`
- `minor_axis_length_raw`
- `main_area_filled`
- `porosity`
- `hu_moment1`
- `moment_of_inertia_xx`
- `moment_of_inertia_yy`
- `moment_of_inertia_xy`
- `type_factor`
- `heywood_circularity_factor`
- `elongation_factor`
- `compactness_factor_ni`

## Stage 2 quality metrics
- `variance_laplacian`
- `tenengrad_variance`
- `energy_laplacian`
- `energy_gradient`

## Stage 2 shape/texture descriptors
Stage 2 generates 20 LBP histogram bins (two radii, ten bins each) by default.
These columns are named sequentially as:

- `lbp_0` through `lbp_19`

Boolean quality guards (`has_black_mask_artifact`, `is_on_crop_boundary`) are
kept alongside these numeric features but are intentionally omitted from the
numeric list above because they serve as filtering flags instead of ML inputs.
