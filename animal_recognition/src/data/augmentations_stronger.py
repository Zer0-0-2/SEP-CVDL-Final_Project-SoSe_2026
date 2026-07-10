import albumentations as A
from albumentations.pytorch import ToTensorV2

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,  # max 10% zoom in/out
                rotate_limit=90,  # pets can be sideways or laying down
                border_mode=0,
                p=0.8,
            ),
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                    A.HueSaturationValue(
                        hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0
                    ),
                    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                ],
                p=0.9,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(distort_limit=0.15, p=1.0),
                    A.GridDistortion(num_steps=4, distort_limit=0.15, p=1.0),
                    A.Perspective(scale=(0.1, 0.2), p=1.0),
                ],
                p=0.8,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.ImageCompression(quality_range=(70, 95), p=1.0),
                ],
                p=0.5,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 5),
                hole_height_range=(16, 48),
                hole_width_range=(16, 48),
                fill=0,
            ),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
            ),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
    )
