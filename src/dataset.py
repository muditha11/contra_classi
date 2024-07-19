import random
import json

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2

from .constants import image_mean, image_std, img_hw


def load_from_json(filename: str):
    """Load json file from file path to a python dictionary"""
    with open(filename, "r") as file:
        return json.load(file)


def load_txt_to_list(path: str):
    """load a .txt file containing triplets to a list of lists"""
    loaded_data = []
    with open(path, "r") as file:
        for line in file:
            inner_list = list(map(int, line.strip().split(",")))
            loaded_data.append(inner_list)
    return loaded_data


class TripletDataset(Dataset):
    """Triplet Dataset definition"""

    def __init__(self, data_dir: str, split: str):

        root = f"{data_dir}/{split}"
        self.triplets = load_txt_to_list(root)

        normalize = v2.Normalize(mean=image_mean, std=image_std)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):

        a, p, n = self.triplets[index]
        anchor = Image.open(a).convert("RGB")
        positive = Image.open(p).convert("RGB")
        negative = Image.open(n).convert("RGB")

        return (
            self.transform(anchor),
            self.transform(positive),
            self.transform(negative),
        )


# TODO : Complete this
class ClassficationDataset(Dataset):

    def __init__(
        self,
        root="./data/oxford-iiit-pet",
        split="train",
    ):

        self.root = root
        self.split = split

        with open(f"{root}/annots/" + split + ".txt", "r") as file:
            read_list = [line.strip() for line in file.readlines()]
        self.imgs = read_list

        self.transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_hw[::-1], scale=[0.8, 1.0]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.Resize(img_hw[::-1]),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_loc = self.imgs[idx]

        label = classes_label_map[
            "_".join(img_loc.split("/")[-1].split(".")[0].split("_")[:-1])
        ]

        image = Image.open(img_loc).convert("RGB")
        if self.split in ["test", "val"]:
            tensor_image = self.transform_test(image)
        else:
            tensor_image = self.transform_train(image)
        return tensor_image, label
