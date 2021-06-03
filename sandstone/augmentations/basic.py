import torch
import torchvision
from sandstone.augmentations.abstract import Abstract_augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ToTensor(Abstract_augmentation):
    '''
    torchvision.transforms.ToTensor wrapper.
    '''

    def __init__(self):
        super(ToTensor, self).__init__()
        self.transform = ToTensorV2()

    def __call__(self, img, additional=None):
        return torch.from_numpy(img).float()

class Permute3d(Abstract_augmentation):
    """Permute tensor (T, C, H, W) ==> (C, T, H, W)"""

    def __init__(self):
        super(Permute3d, self).__init__()

        def permute_3d(tensor):
            return tensor.permute(1, 0, 2, 3)

        self.transform = torchvision.transforms.Lambda(permute_3d)

    def __call__(self, tensor, additional=None):
        return self.transform(tensor)


class ComposeAug(Abstract_augmentation):
    '''
    composes multiple augmentations
    '''

    def __init__(self, augmentations):
        super(ComposeAug, self).__init__()
        self.augmentations = augmentations

    def __call__(self, img, additional=None):
        for transformer in self.augmentations:
            img = transformer(img, additional)

        return img
