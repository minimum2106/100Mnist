import numpy as np
import glob

from torchvision.datasets import VisionDataset
from PIL import Image

class CustomMNIST(VisionDataset):
    
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(self, root, transforms = None, transform = None, target_transform = None):
        super().__init__(root, transforms, transform, target_transform)

        self.images, self.labels = self.__load_data()
        self.train = True

    def __load_data(self):
        images = []
        for f in glob.iglob(f"{self.root}/images"):
            images.append(np.asarray(Image.open(f)))
        labels = np.load(f"{self.root}/labels")

        return images, labels

    def __len__(self):
        return len(self.images)

