import random
import cv2
from torch.utils.data import Dataset
from torchvision import datasets
from base import BaseDataset
from typing import Optional, Callable, List, Tuple


class ClassificationDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        classes_path: str,
        img_wh: Tuple[int, int],
        img_loader: str = 'opencv',  # pil or opencv
        transform: Optional[Callable] = None,  # to samples
        target_transform: Optional[Callable] = None,  # to target
    ):
        super().__init__(root, classes_path, img_wh, img_loader, transform, target_transform)

        self.base = datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        random.shuffle(self.samples)

    def __getitem__(self, idx: int):
        data = self.samples[idx]
        img_path, label = data[0], data[1]
        im = self.img_loader(img_path)

        if self.transform is not None:
            im = self.transform(im)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return im, label

    def __len__(self) -> int:
        return len(self.samples)


if __name__ == '__main__':
    dataset = ClassificationDataset(
        root=r'D:\llf\dataset\mnist\images\train',
        classes_path=r'D:\llf\code\pytorch-module-dataset\test\classes.txt',
        img_wh=(28, 28)
    )
    print(f'root: {dataset.root}')
    print(f'dataset size: {len(dataset)}')
    print(f'nc: {dataset.nc}')
    print(f'classes: {dataset.classes}')
    im, label = dataset[0]
    cv2.imshow('img', im)
    cv2.waitKey(0)
    print(label)
