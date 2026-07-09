import albumentations as A
from albumentations.pytorch import ToTensorV2

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


# vetted using augmentations.ipynb
def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.75, 1.0),
                ratio=(0.8, 1.25),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            # Subtle rotation/shifting
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-30, 30),
                border_mode=0,
                fill=0,
                p=0.7,
            ),
            # Simulate different camera angles
            A.OneOf(
                [
                    A.OpticalDistortion(distort_limit=0.3, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                    A.Perspective(scale=(0.05, 0.1), p=1.0),
                ],
                p=0.5,
            ),
            #  color and lighting stuff
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.HueSaturationValue(
                        hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=20, p=1.0
                    ),
                    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                ],
                p=0.8,
            ),
            # blur
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.ImageCompression(quality_range=(60, 95), p=1.0),
                ],
                p=0.3,
            ),
            # sharpen
            A.Sharpen(alpha=(0.1, 0.2), lightness=(0.5, 1.0), p=0.2),
            # crop out randomly
            A.CoarseDropout(
                num_holes_range=(1, 5),
                hole_height_range=(12, 32),
                hole_width_range=(12, 32),
                fill=0,
                p=0.5,
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
