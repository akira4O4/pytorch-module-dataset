from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import loguru
from torch.utils.data import Dataset
from loguru import logger
from typing import Optional, Callable, List, Tuple


class BaseDataset(Dataset):
    def __init__(
        self,
        root: str,
        classes_path: str,
        img_wh: Tuple[int, int],
        img_loader: str = 'opencv',  # pil or opencv
        transform: Optional[Callable] = None,  # to samples
        target_transform: Optional[Callable] = None,  # to target
    ):
        self.support_img_format = ('.jpg', '.jpeg', '.png')
        self.root = Path(root)
        self.classes_path = Path(classes_path)

        assert img_loader in ['opencv', 'pil'], f'img_loader must be in [opencv,pil]'
        self.img_loader = self.opencv_loader if img_loader == 'opencv' else self.pil_loader

        self.img_wh = img_wh

        self.classes = []
        self.nc = 0

        self.transform = transform
        self.target_transform = target_transform

        self.read_classes_file()

    def read_classes_file(self) -> None:
        if not self.classes_path.exists():
            logger.error(f'Don`t found the classes file: {self.classes_path}')
            return

        with open(self.classes_path, 'r') as f:
            for line in f:
                self.classes.append(str(line.strip()))  # strip() 去除每行末尾的换行符
        self.nc = len(self.classes)

    @staticmethod
    def pil_loader(path: str) -> Image.Image:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    @staticmethod
    def opencv_loader(path: str) -> np.ndarray:
        im = cv2.imread(path)
        if im is None:
            raise FileNotFoundError(f'Don`t open image: {path}')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    @staticmethod
    def pil2cv(im: Image.Image) -> np.ndarray:
        im = np.array(im)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        return im

    @staticmethod
    def cv2pil(im: np.ndarray) -> Image.Image:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        return im


if __name__ == '__main__':
    dataset = BaseDataset('./bbb', r'D:\llf\code\pytorch-module-dataset\test\classes.txt')
    dataset.read_classes_file()
    print(dataset.nc)
    print(dataset.classes)
