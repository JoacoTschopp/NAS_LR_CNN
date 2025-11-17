from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as T
from torchvision import datasets


@dataclass
class TransformConfig:
    img_size: int = 32

    # Geométricas básicas
    use_random_resized_crop: bool = False
    use_random_crop_with_padding: bool = False
    crop_padding: int = 4

    use_random_horizontal_flip: bool = False
    random_horizontal_flip_prob: float = 0.5

    use_random_rotation: bool = False
    rotation_degrees: float = 15.0

    # Políticas avanzadas
    use_autoaugment: bool = False
    use_trivial_augment: bool = False

    # Fotométricas simples
    use_color_jitter: bool = False
    jitter_brightness: float = 0.2
    jitter_contrast: float = 0.2
    jitter_saturation: float = 0.2
    jitter_hue: float = 0.1

    # Regularización en imagen
    use_random_erasing: bool = False
    random_erasing_p: float = 0.25

    normalize: bool = True


def compute_dataset_stats(dataset_root: str):
    cifar10_training = datasets.CIFAR10(dataset_root, train=True, download=True)
    data = cifar10_training.data  # (50000, 32, 32, 3), uint8

    mean = np.mean(data, axis=(0, 1, 2)) / 255.0
    std = np.std(data, axis=(0, 1, 2)) / 255.0

    return mean.tolist(), std.tolist()


def build_transforms(mean, std, config: TransformConfig):
    """
    Construye transformaciones usando la nueva API torchvision.transforms.v2.
    Pensado para CIFAR10 (32x32).
    """
    train_transforms = []

    # -------------------------
    # 1) Geométricas iniciales
    # -------------------------
    if config.use_random_resized_crop:
        train_transforms.append(T.RandomResizedCrop(config.img_size))
    elif config.use_random_crop_with_padding:
        train_transforms.append(
            T.RandomCrop(config.img_size, padding=config.crop_padding)
        )
    else:
        train_transforms.append(T.Resize((config.img_size, config.img_size)))

    if config.use_random_horizontal_flip:
        train_transforms.append(
            T.RandomHorizontalFlip(config.random_horizontal_flip_prob)
        )

    if config.use_random_rotation:
        train_transforms.append(
            T.RandomRotation(degrees=config.rotation_degrees)
        )

    # -------------------------
    # 2) Políticas de AutoAugment
    #    (incluyen color jitter, etc.)
    # -------------------------
    if config.use_autoaugment:
        train_transforms.append(
            T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10)
        )
    elif config.use_trivial_augment:
        train_transforms.append(T.TrivialAugmentWide())

    # -------------------------
    # 3) Fotométricas simples
    #    (solo si NO usamos una política automática)
    # -------------------------
    if config.use_color_jitter and not (config.use_autoaugment or config.use_trivial_augment):
        train_transforms.append(
            T.ColorJitter(
                brightness=config.jitter_brightness,
                contrast=config.jitter_contrast,
                saturation=config.jitter_saturation,
                hue=config.jitter_hue,
            )
        )

    # -------------------------
    # 4) Conversión a tensor + normalización
    # -------------------------
    train_transforms.extend([
        T.ToImage(),                              # PIL/ndarray -> Tensor (C,H,W)
        T.ToDtype(torch.float32, scale=True),     # Escala a [0,1] y dtype float32
    ])

    if config.normalize:
        train_transforms.append(T.Normalize(mean, std))

    # -------------------------
    # 5) Regularización final
    # -------------------------
    if config.use_random_erasing:
        train_transforms.append(
            T.RandomErasing(p=config.random_erasing_p)
        )

    train_transform = T.Compose(train_transforms)

    # Transformaciones de test: sin augmentations pesadas
    test_transform = T.Compose([
        T.Resize((config.img_size, config.img_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean, std) if config.normalize else T.Identity(),
    ])

    return train_transform, test_transform

# ==============================================================================
# VARIANTES DE DATA AUGMENTATION
# ==============================================================================

class config_augmentation:
    

    def __init__(self):
        self.config_sin_augmentation = TransformConfig() # Por defecto solo normaliza

        # Crop+Padding y Horizontal Flip
        self.config_light = TransformConfig(
            img_size=32,
            use_random_resized_crop=False,
            use_random_crop_with_padding=True,
            crop_padding=4,
            use_random_horizontal_flip=True,
            random_horizontal_flip_prob=0.5,
            use_random_rotation=False,
            use_autoaugment=False,
            use_trivial_augment=False,
            use_color_jitter=False,
            use_random_erasing=False,
            normalize=True,
        )

        # AutoAugment + RandomErasing
        self.config_autoaugment = TransformConfig(
            img_size=32,
            use_random_resized_crop=False,
            use_random_crop_with_padding=True,
            crop_padding=4,
            use_random_horizontal_flip=True,
            random_horizontal_flip_prob=0.5,
            use_random_rotation=False,
            use_autoaugment=True,
            use_trivial_augment=False,
            use_color_jitter=False,    # lo hace AutoAugment
            use_random_erasing=True,
            random_erasing_p=0.25,
            normalize=True,
        )

        # Variante geométrica+color jitter
        self.config_geometrica = TransformConfig(
            img_size=32,
            use_random_resized_crop=False,
            use_random_crop_with_padding=True,
            crop_padding=4,
            use_random_horizontal_flip=True,
            random_horizontal_flip_prob=0.5,
            use_random_rotation=True,
            rotation_degrees=15,
            use_autoaugment=False,
            use_trivial_augment=False,
            use_color_jitter=True,
            jitter_brightness=0.3,
            jitter_contrast=0.3,
            jitter_saturation=0.3,
            jitter_hue=0.1,
            use_random_erasing=True,
            random_erasing_p=0.15,
            normalize=True,
        )