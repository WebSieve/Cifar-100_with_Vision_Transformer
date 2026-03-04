import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional

from .config import TrainingConfig


def get_transforms(is_training: bool = True, img_size: int = 32) -> transforms.Compose:
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    if is_training:
        return transforms.Compose(
            [
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.25),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


def get_cifar100_loaders(
    config: TrainingConfig, img_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    train_transform = get_transforms(is_training=True, img_size=img_size)
    test_transform = get_transforms(is_training=False, img_size=img_size)

    train_dataset = torchvision.datasets.CIFAR100(
        root=config.data_dir,
        train=True,
        transform=train_transform,
        download=True,
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=config.data_dir,
        train=False,
        transform=test_transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, test_loader


class Mixup:
    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha

    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        else:
            lam = 1.0

        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)

        mixed_images = lam * images + (1 - lam) * images[index]

        return mixed_images, labels, labels[index], lam


class CutMix:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        else:
            lam = 1.0

        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)

        _, _, H, W = images.shape

        cut_ratio = (1 - lam) ** 0.5
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        cy = torch.randint(H, (1,)).item()
        cx = torch.randint(W, (1,)).item()

        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)

        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)

        return mixed_images, labels, labels[index], lam
