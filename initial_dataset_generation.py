import gzip
import numpy as np
from typing import Sequence, Union
import argparse
import os

import torch
from torch import nn

import torchvision
from torchvision.datasets import MNIST
from torchvision.utils import save_image

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

IMG_SIZE = 28


class RandomCenterCrop(nn.Module):
    def __init__(
        self,
        min_cropped_img_size: Union[int, Sequence[int]],
        remain_shape=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if type(min_cropped_img_size) == int:
            self.min_cropped_img_size = (min_cropped_img_size, min_cropped_img_size)
        else:
            self.min_cropped_img_size = min_cropped_img_size

        self.remain_shape = remain_shape

    def forward(self, img):
        if (img.shape[1] < self.min_cropped_img_size[0]) | (
            img.shape[2] < self.min_cropped_img_size[1]
        ):
            return Warning(
                "min_cropped_size is bigger than the image's original dimension"
            )

        random_crop_size_h = np.random.randint(
            self.min_cropped_img_size[0], img.shape[1]
        )
        random_crop_size_w = np.random.randint(
            self.min_cropped_img_size[1], img.shape[2]
        )
        transformed_img = v2.CenterCrop([random_crop_size_h, random_crop_size_w])(img)

        if self.remain_shape:
            img_zeros = torch.zeros((img.shape[1:]))
            start_w = (img.shape[2] - transformed_img.shape[2]) // 2
            end_w = transformed_img.shape[2] + start_w
            start_h = (img.shape[1] - transformed_img.shape[1]) // 2
            end_h = transformed_img.shape[1] + start_h

            img_zeros[start_h:end_h, start_w:end_w] = transformed_img
            transformed_img = img_zeros

        return transformed_img


def get_images(gzip_file, num_image):
    f = gzip.open(gzip_file)
    f.read(16)

    buf = f.read(IMG_SIZE * IMG_SIZE * num_image)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_image, IMG_SIZE, IMG_SIZE, 1)

    return data


def get_labels(gzip_file, num_label):
    f = gzip.open(gzip_file)
    f.read(8)

    buf = f.read(num_label)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    return data


def random_img_with_label(images, labels, label, size=10):
    indices = np.where(labels == label)[0]
    chosen_indices = np.random.choice(indices, size=size)
    images = images[torch.from_numpy(chosen_indices)]

    return (images, labels[chosen_indices])


def apply_transformation(
    img, transformation, img_counter, num_generation=1, foldername="", save=False
):
    transformed_imgs = [transformation(img) for _ in range(num_generation)]
    if save:
        for transformed_img in transformed_imgs:
            path = os.path.join(foldername, "images", f"{img_counter}.jpg")
            save_image(transformed_img / 255, path)

            img_counter += 1

    return transformed_imgs, img_counter


parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_sample", type=int, default=10, help="number of orginal samples in each class"
)
parser.add_argument(
    "--keep_samples",
    type=bool,
    default=True,
    help="keep copies of orginal image in the generated dataset or not",
)
parser.add_argument(
    "--folder_name", type=str, default="dataset", help="dataset's folder's name"
)
parser.add_argument(
    "--num_generation_per_transformation",
    type=int,
    default=10,
    help="number of image generated for each transformation",
)

parser.add_argument(
    "--crop", type=bool, default=True, help="augmentation: add crop or not"
)
parser.add_argument(
    "--min_cropped_img_size",
    type=int,
    default=24,
    help="minimum dimension's length of the cropped image",
)

parser.add_argument(
    "--zoom", type=bool, default=True, help="augmentation: add crop or not"
)
parser.add_argument(
    "--zoom_min_size_range",
    type=float,
    default=1,
    help="minimum size range of RandomZoomOut",
)
parser.add_argument(
    "--zoom_max_size_range",
    type=float,
    default=2,
    help="minimum size range of RandomZoomOut",
)
parser.add_argument(
    "--rotate", type=bool, default=True, help="augmentation: add crop or not"
)


def check_in_range(input):
    input = int(input)
    if (input > 360) or (input < 0):
        raise argparse.ArgumentTypeError("degree is out of 0 - 360 degrees")

    return input


parser.add_argument(
    "--rotation_degrees",
    type=check_in_range,
    default=45,
    help="minimum dimension's length of the cropped image",
)
parser.add_argument(
    "--translate",
    type=bool,
    default=True,
    help="augmentation: add vertical / horizontal translation to the image",
)

parser.add_argument(
    "--horizontal_translate",
    type=float,
    default=0.5,
    help="maximum absolute fraction for horizontal translations",
)

parser.add_argument(
    "--vertical_translate",
    type=float,
    default=0.5,
    help="maximum absolute fraction for vertical translations",
)

args = parser.parse_args()


# randomly pick 10 images each class
# train_images = get_images("train-images-idx3-ubyte.gz", 60000)
# train_labels = get_labels("train-labels-idx1-ubyte.gz", 60000)

train_datasets = MNIST(
    "./",
    train=True,
    transform=v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()]),
    download=True,
)
train_images = train_datasets.data
train_labels = train_datasets.targets

all_labels = np.array([])

os.makedirs(args.folder_name, exist_ok=True)
for i in range(10):
    img_counter = 0
    images, labels = random_img_with_label(
        train_images, train_labels, i, size=args.num_sample
    )

    os.makedirs(f"{args.folder_name}/images", exist_ok=True)
    if args.keep_samples:
        for img in images:
            path = os.path.join(args.folder_name, "images", f"{img_counter}.jpg")
            save_image(img / 255, path)
            img_counter += 1

        all_labels = np.hstack((all_labels, labels))

    transformations = []

    if args.crop:
        transformations.append(RandomCenterCrop(args.min_cropped_img_size))

    if args.zoom:
        transformations.append(
            v2.RandomZoomOut(
                fill=0, side_range=(args.zoom_min_size_range, args.zoom_max_size_range)
            )
        )

    if args.rotate:
        transformations.append(v2.RandomRotation(args.rotation_degrees))

    if args.translate:
        v2.RandomAffine(
            degrees=0,
            translate=(args.horizontal_translate, args.vertical_translate),
            fill=0,
        )

    for transformation in transformations:
        for img in images:
            _, img_counter = apply_transformation(
                img.view(1, IMG_SIZE, IMG_SIZE),
                transformation,
                img_counter,
                num_generation=args.num_generation_per_transformation,
                foldername=args.folder_name,
                save=True,
            )

        all_labels = np.hstack(
            (all_labels, np.repeat(labels, args.num_generation_per_transformation))
        )

with open(f"{args.folder_name}/labels.npy", "wb") as f:
    np.save(f, all_labels)
