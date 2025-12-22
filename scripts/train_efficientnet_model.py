"""
Train an EfficientNet-B0 classifier on Stage 2 crop images with strong augmentation.

This script fine-tunes an ImageNet-initialized EfficientNet-B0 with mixup and
cutout/blur/noise augmentations. It mirrors the configuration-driven approach
used by the RandomForest trainer so the resulting checkpoint can be selected via
`stage_03_filtering_and_classification.model_type` in the pipeline config.
"""

import argparse
import logging
import os
import random
import sys
from typing import Tuple

# Ensure project root is on the Python path for module resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.utils import load_config_once, setup_logging, get_models_dir, get_training_data_dir
from src.utils.common_exceptions import ConfigError

logger = logging.getLogger(__name__)


class AddGaussianNoise(object):
    """Transform that adds Gaussian noise to an image tensor."""

    def __init__(self, mean=0.0, std=0.02):
        """Store noise statistics for the transform."""
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """Apply Gaussian noise to the provided tensor."""
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


class MixupWrapper:
    """Wrapper for applying mixup augmentation during training."""

    def __init__(self, alpha: float):
        """Configure the Beta distribution strength for mixing samples."""
        self.alpha = alpha

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Mix input batch and labels using the configured mixup alpha."""
        if self.alpha <= 0:
            return inputs, targets, targets, 1.0
        lam = np.random.beta(self.alpha, self.alpha)
        index = torch.randperm(inputs.size(0), device=inputs.device)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        targets_a, targets_b = targets, targets[index]
        return mixed_inputs, targets_a, targets_b, lam


def _get_normalization(weights: EfficientNet_B0_Weights) -> Tuple[list, list]:
    """Return normalization statistics for the provided weights.

    The torchvision weights metadata changed in recent releases (mean/std were
    removed from ``weights.meta``). This helper first tries the legacy
    ``meta`` mapping and then falls back to the attributes exposed by the
    transforms preset. If neither is available, a clear error is raised so
    callers don't see an unexpected ``KeyError``.
    """

    meta = getattr(weights, "meta", {}) or {}
    mean, std = meta.get("mean"), meta.get("std")
    if mean and std:
        return mean, std

    preset = weights.transforms()
    if hasattr(preset, "mean") and hasattr(preset, "std"):
        return preset.mean, preset.std

    raise RuntimeError("Could not determine normalization statistics from the provided weights.")


def build_transforms(input_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create training and validation transform pipelines for EfficientNet."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    mean, std = _get_normalization(weights)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.02),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)),
            normalize,
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize(input_size + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_tfms, val_tfms


def build_model(num_classes: int) -> nn.Module:
    """Initialize EfficientNet-B0 with a classifier head sized for the dataset."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, mixup: MixupWrapper, device: torch.device) -> Tuple[float, float]:
    """Train the model for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        mixed_inputs, targets_a, targets_b, lam = mixup(inputs, targets)

        optimizer.zero_grad()
        outputs = model(mixed_inputs)
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (lam * preds.eq(targets_a).sum().item()) + ((1 - lam) * preds.eq(targets_b).sum().item())
        total += inputs.size(0)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device: torch.device) -> Tuple[float, float]:
    """Evaluate the model on the validation set and return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(targets).sum().item()
            total += inputs.size(0)

    return total_loss / total, correct / total


def parse_args():
    """Parse command-line arguments for EfficientNet training."""
    parser = argparse.ArgumentParser(description="Train an EfficientNet-B0 classifier with strong augmentation.")
    parser.add_argument('--config', default='config/pipeline_config.yml', help='Path to the pipeline config file.')
    parser.add_argument(
        '--dataset-dir',
        default=None,
        help='Directory with class subfolders for training (defaults to config-specified training_data path).',
    )
    parser.add_argument('--epochs', type=int, default=None, help='Override training epochs (defaults from config).')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size (defaults from config).')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--mixup-alpha', type=float, default=0.4)
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help="Device to train on: 'cuda', 'cpu', or 'auto' (prefer CUDA if available).",
    )
    return parser.parse_args()


def main():
    """Train EfficientNet-B0 using curated crops and save the best checkpoint."""
    args = parse_args()
    config = load_config_once(args.config)
    s3_config = config.get('stage_03_filtering_and_classification', {})
    eff_cfg = s3_config.get('efficient_net_model', {})
    if not eff_cfg:
        raise ConfigError("Add an 'efficient_net_model' section to the Stage 3 config to control filenames and labels.")

    checkpoint_filename = eff_cfg.get('checkpoint_filename', 'efficientnet_b0_best.pt')
    input_size = int(eff_cfg.get('input_size', 224))
    positive_label = eff_cfg.get('positive_label', 'good')
    training_cfg = (
        s3_config.get('training_params', {})
        .get('efficient_net_training', {})
    )
    default_epochs = int(training_cfg.get('default_epochs', 15))
    default_batch_size = int(training_cfg.get('default_batch_size', 32))
    dataset_subdir = training_cfg.get('dataset_subdir', '') or ''
    models_dir = get_models_dir(config)
    os.makedirs(models_dir, exist_ok=True)

    # Reuse the same curated crop layout as the RandomForest path: data/training_data/{good,bad}
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        base_training_dir = get_training_data_dir(config)
        dataset_dir = os.path.join(base_training_dir, dataset_subdir) if dataset_subdir else base_training_dir
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Training dataset directory not found: {dataset_dir}")

    setup_logging(config)
    logger.info("Starting EfficientNet training with dataset at %s", dataset_dir)

    train_tfms, val_tfms = build_transforms(input_size)
    full_dataset = ImageFolder(dataset_dir, transform=train_tfms)
    class_to_idx = full_dataset.class_to_idx
    num_classes = len(class_to_idx)
    if num_classes < 2:
        raise ConfigError("Training data must contain at least two classes.")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_tfms

    epochs = args.epochs or default_epochs
    batch_size = args.batch_size or default_batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Rerun with --device cpu or install GPU support.")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    mixup = MixupWrapper(alpha=args.mixup_alpha)

    best_val_acc = 0.0
    best_checkpoint = None

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, mixup, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        logger.info(
            "Epoch %d/%d -> train loss: %.4f acc: %.3f | val loss: %.4f acc: %.3f",
            epoch + 1,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint = {
                'model_state_dict': model.state_dict(),
                'class_to_idx': class_to_idx,
                'input_size': input_size,
                'positive_label': positive_label,
                'val_acc': val_acc,
            }

    if best_checkpoint is None:
        raise RuntimeError("Training did not produce a valid checkpoint. Check the dataset and parameters.")

    checkpoint_path = os.path.join(models_dir, checkpoint_filename)
    torch.save(best_checkpoint, checkpoint_path)
    logger.info("Saved best EfficientNet checkpoint to %s (val acc %.3f)", checkpoint_path, best_val_acc)


if __name__ == "__main__":
    main()
