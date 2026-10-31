"""Common interface for image encoders that preserve spatial information."""

from abc import ABC, abstractmethod
import torch
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

def spatial_pair(value):
    pair = (value, value) if isinstance(value, int) else tuple(value)
    if len(pair) != 2 or any(int(v) != v or v <= 0 for v in pair):
        raise ValueError(f"Expected a positive (height, width), got {value}")
    return tuple(int(v) for v in pair)


class SpatialEncoder(nn.Module, ABC):
    latent_ndim = 2
    def __init__(self, name, emb_dim, grid_size, input_size, mean, std,
                 interpolation=InterpolationMode.BICUBIC):
        super().__init__()
        self.name = name
        self.emb_dim = int(emb_dim)
        if self.emb_dim <= 0:
            raise ValueError("emb_dim must be positive")
        self.grid_size = spatial_pair(grid_size)
        self.input_size = spatial_pair(input_size)
        self.interpolation = interpolation
        self.register_buffer("image_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    @property
    def num_patches(self):
        return self.grid_size[0] * self.grid_size[1]

    def preprocess(self, images):
        images = (images + 1.0) / 2.0
        images = TF.resize(images, list(self.input_size),
                           interpolation=self.interpolation, antialias=True)
        return (images - self.image_mean.to(images)) / self.image_std.to(images)

    def forward(self, images):
        tokens = self.forward_tokens(self.preprocess(images))
        return tokens

    @abstractmethod
    def forward_tokens(self, images):
        """Return spatial tokens from images already preprocessed by this encoder."""


def encoder_metadata(encoder):
    """Read metadata through a distributed wrapper without bypassing its forward."""
    while hasattr(encoder, "module"):
        encoder = encoder.module
    return encoder
