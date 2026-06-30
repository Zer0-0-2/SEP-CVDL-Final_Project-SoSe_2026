import albumentations as A
from albumentations.pytorch import ToTensorV2

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.6, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, 32), hole_width_range=(1, 32), p=0.3),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])
