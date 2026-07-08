import albumentations as A
from albumentations.pytorch import ToTensorV2

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


# similar to main augmentations.py but less agressive. 
# For testing rn. Suprressed types so vscode doesn't complain, should work though
def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose(  # type: ignore
        [  # type: ignore
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.85, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.7),
            A.OneOf(
                [  # type: ignore
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.ImageCompression(quality_range=(50, 85)),
                ],
                p=0.4,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4), hole_height_range=(1, 16), hole_width_range=(1, 16), p=0.2
            ),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [  # type: ignore
            # Preserves aspect ratio by scaling the longest side to 224, then padding the rest
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
