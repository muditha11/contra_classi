"""
--- Model definition ---
available classes:
    Encoder
    Triplet model
    Classification model
"""


import torch.nn as nn
import torch
import timm


class Encoder(nn.Module):
    """Encoder class definition"""

    def __init__(
        self, base_model: str, version: str, device: int = 0, checkpoint: str = None
    ):
        super().__init__()
        self.base_model = base_model
        self.version = version
        self.device = device

        self.model = timm.create_model(
            f"{base_model}.{version}", pretrained=True, num_classes=2
        )
        if base_model == "convnext_tiny":
            self.model.head.fc = nn.Identity()

        if base_model == "vit_base_patch16_224":
            self.model.head = nn.Identity()

        if checkpoint is not None:
            ckpt = torch.load(checkpoint)
            self.model.load_state_dict(ckpt["state_dict"])
            print("Loaded model weights successfully!!")

        self.to(device)

    def save_encoder(self, path: str):
        """saves the encoder to input path"""
        checkpoint = {
            "state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, f"{path}/{self.base_model}_encoder.pt")

    def forward(self, x):
        """forward method"""
        x = x.to(self.device)
        y = self.model(x)
        return y


class TripletModel(nn.Module):
    """Triplet model definition"""

    def __init__(self, base_model: str, version: str, device: int = 0, weights=None):
        super().__init__()
        self.base_model = base_model
        self.encoder = Encoder(base_model, version, device, weights)
        self.to(device)
        self.device = device

    def save_encoder(self, path):
        """saves the encoder to input path

        Args:
            path (str): where to save the model to
        """
        self.encoder.save_encoder(path)

    def forward(self, batch):
        """forward method"""

        anchor, positive, negative = batch
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        anchor_out = self.encoder(anchor)
        positive_out = self.encoder(positive)
        negative_out = self.encoder(negative)
        return anchor_out, positive_out, negative_out


class ClassificationModel(nn.Module):
    """Classification model definition"""

    def __init__(self, base_model: str, version: str, classes:int,device: int = 0, weights=None):
        super().__init__()
        self.base_model = base_model
        self.encoder = Encoder(base_model, version, device, weights)
        self.decoder = nn.Linear(768,classes)
        self.device = device
        self.to(device)

    def save_encoder(self, path):
        """saves the encoder to input path

        Args:
            path (str): where to save the model to
        """
        self.encoder.save_encoder(path)

    def forward(self, batch):
        """forward method"""

        x = batch[0]
        x = x.to(self.device)
        y = self.encoder(x).flatten(start_dim=1)
        y = self.decoder(y)
        return y
