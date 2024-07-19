import random
import json

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2

from .constants import image_mean, image_std


class TripletDataset(Dataset):
    """Triplet Dataset definition"""

    def __init__(self, data_dir: str, split: str):

        self.df = pd.read_csv(f"{data_dir}/{split}.csv")
        normalize = v2.Normalize(mean=image_mean, std=image_std)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                RandomRotate90(),
                normalize,
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        a, p, n = self.df.iloc[index]
        A = Image.open(a).convert("RGB")
        P = Image.open(p).convert("RGB")
        N = Image.open(n).convert("RGB")

        return self.transform(A), self.transform(P), self.transform(N)


# TODO: Classfication dataset
