# Training the EfficientNet-B0 stage 3 classifier

This guide shows how to train the CNN alternative to the RandomForest stage 3 classifier and how to toggle between them in the pipeline configuration. It also covers practical notes for CPU-only versus GPU-enabled machines.

## 1. Prepare the dataset

Reuse the same curated crops used by the RandomForest classifier. By default the trainer looks under `data/training_data/` with two class folders: `good/` and `bad/`.

```
data/
  training_data/
    good/
      img_001.png
      img_002.png
    bad/
      img_101.png
      img_102.png
```

* The folder names become class labels. A minimum of two classes is required.
* Images are standard RGB crops (the same as produced by stage 2). The script will apply all normalization and augmentation.

## 2. Configure the pipeline to recognize the EfficientNet model

Ensure the Stage 3 section of `config/pipeline_config.yml` contains the `efficient_net_model` block. Example:

```yaml
stage_03_filtering_and_classification:
  model_type: efficient_net_model  # switch to random_forest_model to revert
  efficient_net_model:
    checkpoint_filename: efficientnet_b0_best.pt
    input_size: 224
```

`checkpoint_filename` is where the trainer will save the best model (inside `models/`).

## 3. Train the model

From the repository root, run:

```
python scripts/train_efficientnet_model.py \
  --config config/pipeline_config.yml \
  --epochs 20 \
  --batch-size 32
```

Key flags:

* `--dataset-dir`: folder containing the class subfolders. Defaults to `data/training_data` if omitted.
* `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--val-split`, `--mixup-alpha`: optional training hyperparameters.
* `--device`: `auto` (default, prefers CUDA if available), `cuda`, or `cpu`.

The script saves the best-performing checkpoint (by validation accuracy) to `models/<checkpoint_filename>`.

## 4. CPU vs. GPU considerations

* **GPU available**: training is significantly faster. Leave `--device auto` (default) or explicitly set `--device cuda`. Ensure CUDA is available (PyTorch installed with CUDA). Larger batch sizes (32–64) usually work well.
* **CPU only**: set `--device cpu` to avoid CUDA checks. Training will be slower; consider reducing `--batch-size` (e.g., 8–16) or epochs to shorten runs. All augmentations still run on CPU.
* Inference in the pipeline will also use CUDA when available; otherwise it will fall back to CPU automatically.

## 5. Switching between models at runtime

After training, set `stage_03_filtering_and_classification.model_type` to `efficient_net_model` to use the CNN. To revert, set it to `random_forest_model`; no other changes are required.
