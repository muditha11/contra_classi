"""
--- Dataset definition ---
available classes:
    Triplet Dataset
    Classfication Dataset
"""

import os
import random
import json

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2

from .constants import image_mean, image_std, img_hw


def pick_random_pos(p_list, anchor):
    while True:
        p = random.choice(p_list)
        if p != anchor:
            return p


def load_from_json(filename: str):
    """Load json file from file path to a python dictionary"""
    with open(filename, "r") as file:
        return json.load(file)


def load_txt_to_list(path: str):
    """load a .txt file containing triplets to a list of lists"""
    loaded_data = []
    with open(path, "r") as file:
        for line in file:
            inner_list = line.strip().split(" ")
            loaded_data.append(inner_list)
    return loaded_data


class TripletDataset(Dataset):
    """Triplet Dataset definition"""

    def __init__(self, data_dir: str, split: str):

        root = f"{data_dir}/{split}.txt"
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

        img_root = "/".join(a.split("/")[:-2])

        pos_dir = os.listdir(f"{img_root}/{p}")
        neg_dir = os.listdir(f"{img_root}/{n}")

        p_im = pick_random_pos(pos_dir, a)
        n_im = random.choice(neg_dir)

        anchor = Image.open(a).convert("RGB")
        positive = Image.open(f"{img_root}/{p}/{p_im}").convert("RGB")
        negative = Image.open(f"{img_root}/{n}/{n_im}").convert("RGB")

        return (
            self.transform(anchor),
            self.transform(positive),
            self.transform(negative),
        )


class ClassficationDataset(Dataset):
    """Classification dataset definition"""

    def __init__(
        self,
        root: str,
        split: str,
    ):

        self.root = root
        self.split = split
        if "pet" in self.split:
            self.classes_label_map = load_from_json(f"{root}/pet_labels.json")
        else:
            self.classes_label_map = load_from_json(f"{root}/micro_labels.json")

        with open(f"{root}/" + split + ".txt", "r") as file:
            read_list = [line.strip() for line in file.readlines()]
        self.imgs = read_list

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
        return len(self.imgs)

    def __getitem__(self, idx):
        img_loc = self.imgs[idx]

        if "pet" in self.split:
            label = self.classes_label_map[
                "_".join(img_loc.split("/")[-1].split(".")[0].split("_")[:-1])
            ]

        else:
            label = self.classes_label_map[img_loc.split("/")[-2]]

        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)

        return tensor_image, label
