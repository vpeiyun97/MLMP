"""Train a basic U-Net model for image segmentation.

This script demonstrates a complete training pipeline that covers:

* Reading images and masks from disk.
* Splitting the dataset into train/validation/test subsets.
* Building a vanilla U-Net model in PyTorch.
* Training with mini-batches and tracking metrics.
* Saving the best model checkpoint and final model weights.

The expected dataset directory structure is::

    data_root/
        images/
            example_001.png
            ...
        masks/
            example_001.png
            ...

Each mask should be a single-channel image where foreground pixels are
represented with non-zero values. Images and masks are matched by
filename. You can adjust the file extensions using the ``--image-ext``
and ``--mask-ext`` command-line arguments.

Example usage::

    python unet_training.py \
        --data-root /path/to/your/dataset \
        --epochs 25 \
        --batch-size 4 \
        --output-dir runs/unet_example

The script is intentionally self-contained so it can be used as a
starting point for small experiments or teaching purposes.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


# ---------------------------------------------------------------------------
# Reproducibility helpers


def set_seed(seed: int = 42) -> None:
    """Set seeds for common random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset definition


class SegmentationDataset(Dataset):
    """Pair images with their corresponding segmentation masks."""

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        image_transform: transforms.Compose | None = None,
        mask_transform: transforms.Compose | None = None,
    ) -> None:
        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of images and masks must be equal.")

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        # Ensure the mask is binary (0 or 1) for BCEWithLogitsLoss
        mask = (mask > 0.5).float()
        return image, mask


def list_image_pairs(
    data_root: Path, image_ext: str = "png", mask_ext: str | None = None
) -> Tuple[List[Path], List[Path]]:
    """List all image/mask pairs in the dataset directory."""

    mask_ext = mask_ext or image_ext
    image_dir = data_root / "images"
    mask_dir = data_root / "masks"

    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            "Dataset directory must contain 'images' and 'masks' subdirectories."
        )

    image_paths = sorted(image_dir.glob(f"*.{image_ext}"))
    mask_paths = sorted(mask_dir.glob(f"*.{mask_ext}"))

    if not image_paths:
        raise FileNotFoundError(
            f"No image files with extension .{image_ext} found in {image_dir}."
        )

    masks_by_name = {mask_path.stem: mask_path for mask_path in mask_paths}
    paired_images: List[Path] = []
    paired_masks: List[Path] = []

    for image_path in image_paths:
        mask_path = masks_by_name.get(image_path.stem)
        if mask_path is None:
            raise FileNotFoundError(
                f"Mask file for image '{image_path.name}' not found in {mask_dir}."
            )
        paired_images.append(image_path)
        paired_masks.append(mask_path)

    return paired_images, paired_masks


def build_datasets(
    data_root: Path,
    image_size: int,
    val_split: float,
    test_split: float,
    image_ext: str = "png",
    mask_ext: str | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Create train/val/test datasets with matching transforms."""

    image_paths, mask_paths = list_image_pairs(data_root, image_ext, mask_ext)

    base_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    dataset = SegmentationDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        image_transform=base_transform,
        mask_transform=transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        ),
    )

    dataset_size = len(dataset)
    test_len = int(dataset_size * test_split)
    val_len = int(dataset_size * val_split)
    train_len = dataset_size - val_len - test_len
    if train_len <= 0:
        raise ValueError("Dataset split results in empty training set. Adjust splits.")

    return random_split(dataset, [train_len, val_len, test_len])


# ---------------------------------------------------------------------------
# U-Net architecture


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = True) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ---------------------------------------------------------------------------
# Training utilities


@dataclass
class TrainState:
    epoch: int
    loss: float
    val_loss: float
    dice: float


def dice_coefficient(outputs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> float:
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > 0.5).float()
    intersection = (outputs * targets).sum(dim=(1, 2, 3))
    union = outputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = ((2.0 * intersection + eps) / (union + eps)).mean()
    return dice.item()


def run_epoch(
    model: nn.Module,
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool = True,
) -> Tuple[float, float]:
    total_loss = 0.0
    total_dice = 0.0
    model.train(train)

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_dice += dice_coefficient(outputs, masks) * images.size(0)

    dataset_size = len(dataloader.dataset)
    return total_loss / dataset_size, total_dice / dataset_size


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_dir: Path,
    filename: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a basic U-Net model")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to dataset root")
    parser.add_argument("--image-ext", type=str, default="png", help="Image file extension")
    parser.add_argument("--mask-ext", type=str, default=None, help="Mask file extension")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--image-size", type=int, default=256, help="Input resize dimension")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/unet"), help="Directory to save models"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of DataLoader worker processes"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, val_dataset, test_dataset = build_datasets(
        data_root=args.data_root,
        image_size=args.image_size,
        val_split=args.val_split,
        test_split=args.test_split,
        image_ext=args.image_ext,
        mask_ext=args.mask_ext,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = UNet(n_channels=3, n_classes=1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    history: List[TrainState] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_dice = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        history.append(TrainState(epoch, train_loss, val_loss, val_dice))
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, args.output_dir, "best_model.pt")

    # Save the final model weights regardless of validation performance
    save_checkpoint(model, optimizer, args.epochs, args.output_dir, "last_model.pt")

    # Evaluate on the holdout test set
    test_loss, test_dice = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )
    print(f"Test loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}")

    # Persist the training history for later analysis
    history_path = args.output_dir / "history.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as file:
        file.write("epoch,train_loss,val_loss,val_dice\n")
        for state in history:
            file.write(f"{state.epoch},{state.loss:.6f},{state.val_loss:.6f},{state.dice:.6f}\n")

    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()

