from __future__ import annotations

from collections.abc import Sequence

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ZERO_ONE_MEAN = (0.0, 0.0, 0.0)
ZERO_ONE_STD = (1.0, 1.0, 1.0)


def _height_width(image_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        return image_size, image_size
    if len(image_size) != 2:
        raise ValueError(f"image_size must contain [height, width], got {image_size}")
    return int(image_size[0]), int(image_size[1])


def build_unet_train_transform(
    image_size: int | Sequence[int] = (256, 256),
    use_pretrained: bool = False,
) -> A.Compose:
    height, width = _height_width(image_size)
    mean = IMAGENET_MEAN if use_pretrained else ZERO_ONE_MEAN
    std = IMAGENET_STD if use_pretrained else ZERO_ONE_STD
    return A.Compose(
        [
            A.Resize(height, width, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=30,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def build_unet_eval_transform(
    image_size: int | Sequence[int] = (256, 256),
    use_pretrained: bool = False,
) -> A.Compose:
    height, width = _height_width(image_size)
    mean = IMAGENET_MEAN if use_pretrained else ZERO_ONE_MEAN
    std = IMAGENET_STD if use_pretrained else ZERO_ONE_STD
    return A.Compose(
        [
            A.Resize(height, width, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def build_train_transform(image_size: int | Sequence[int]) -> A.Compose:
    height, width = _height_width(image_size)
    return A.Compose(
        [
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=30,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=0.4),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_eval_transform(image_size: int | Sequence[int]) -> A.Compose:
    height, width = _height_width(image_size)
    return A.Compose(
        [
            A.Resize(height, width),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_segformer_train_transform(image_size: int | Sequence[int] = (512, 512)) -> A.Compose:
    height, width = _height_width(image_size)
    return A.Compose(
        [
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=30,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_segformer_eval_transform(image_size: int | Sequence[int] = (512, 512)) -> A.Compose:
    height, width = _height_width(image_size)
    return A.Compose(
        [
            A.Resize(height, width),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
