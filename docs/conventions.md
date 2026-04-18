# Coding Conventions

This document defines the naming and style conventions for the
`well_plate_analyzer` codebase.  All new code and refactoring work
**must** follow these conventions.  PRs that introduce violations
should be rejected at review time.

---

## 1. Image array variables

| Rule | Example |
|------|---------|
| Always `image_{adjective/type}` | `image_gray`, `image_color`, `image_float`, `image_uint8` |
| Never use `img_` prefix | ~~`img_gray`~~, ~~`gray_img`~~ |
| Never use `{adjective}_image` suffix | ~~`gray_image`~~ |

```python
# Good
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Bad
gray_img = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
```

---

## 2. Configuration dictionaries

| Rule | Example |
|------|---------|
| Always `{scope}_config` | `mask_config`, `crop_config`, `well_analysis_config` |
| Never use `cfg` abbreviation | ~~`cfg_well_id`~~, ~~`cfg`~~ |
| Local aliases must also use the full form | `well_id_config = config_well_analysis.get(...)` |

```python
# Good
def mask_well_area(image_gray, mask_config, system_config):
    ...

# Bad
def mask_well_area(image_gray, mask_cfg, system_params):
    ...
```

---

## 3. Boolean mask variables

| Rule | Example |
|------|---------|
| Always `{name}_mask` — no `_bool` suffix | `binary_mask`, `body_mask`, `filled_mask` |
| The `_bool` suffix is redundant; the type annotation conveys the dtype | ~~`raw_body_mask_bool`~~ |

```python
# Good
initial_mask = generate_binary_mask_from_float_image(image_float, params)
cleaned_mask = skimage.morphology.remove_small_objects(initial_mask, ...)

# Bad
main_binary_mask_initial_bool = generate_binary_mask_from_float_image(...)
main_cleaned_mask_bool = skimage.morphology.remove_small_objects(...)
```

---

## 4. Private helpers

| Rule | Example |
|------|---------|
| All module-private functions must start with `_` | `_require_cv2`, `_load_image_with_pil` |
| Public API functions must **not** start with `_` | `load_image`, `apply_mask` |

```python
# Good
def _compute_enclosing_circle(contour):  # internal helper
    ...

def compute_aabb(contour):               # public
    ...
```

---

## 5. YAML config parameter dicts (assembled from config file)

| Rule | Example |
|------|---------|
| When a function assembles a sub-dict from the YAML tree, name it `params` | `threshold_params`, `segmentation_params` |
| Do not use `config` for these assembled sub-dicts | ~~`threshold_config`~~ when it holds only params, not the full config |

---

## 6. Magic numbers and string literals

All numeric thresholds, device IDs, filenames, and tuneable values
**must** be defined in `src/utils/constants.py`.  Inline literals are
not permitted in business logic.

```python
# Good
from src.utils.constants import MORPH_KERNEL_SIZE_LARGE
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (MORPH_KERNEL_SIZE_LARGE, MORPH_KERNEL_SIZE_LARGE))

# Bad
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
```

---

## 7. String formatting

Use f-strings exclusively.  `%`-style and `.format()` strings are not
permitted in new code; existing uses should be migrated on touch.

```python
# Good
logger.debug(f"Loaded image {image_path!r} in {mode!r} mode.")

# Bad
logger.debug("Loaded image %s in %s mode.", image_path, mode)
```

---

## 8. Test placement

Self-contained tests inside `if __name__ == "__main__":` blocks are
**not** permitted in `src/`.  All tests must live in `tests/` as
properly named pytest functions (`test_*`).
