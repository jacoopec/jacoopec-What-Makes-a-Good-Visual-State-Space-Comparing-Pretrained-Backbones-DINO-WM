import torch
from torchvision.transforms import InterpolationMode

from models.encoder.base import SpatialEncoder, spatial_pair


class DinoV2Encoder(SpatialEncoder):
    def __init__(self, name="dinov2_vits14", feature_key="x_norm_patchtokens",
                 input_size=196, layer=None, normalization="imagenet"):
        if normalization not in ("imagenet", "legacy"):
            raise ValueError("normalization must be 'imagenet' or 'legacy'")
        model = torch.hub.load("facebookresearch/dinov2:b48308a", name)
        size = spatial_pair(input_size)
        patch = spatial_pair(model.patch_size)
        if any(s % p for s, p in zip(size, patch)):
            raise ValueError("DINO input_size must be divisible by its patch size")
        if layer is not None and not 0 <= layer < len(model.blocks):
            raise ValueError("layer must be a zero-based DINO transformer block index")
        mean, std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if normalization == "legacy":
            # Preserve the original pipeline's [-1, 1] features for old runs.
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        super().__init__(name, model.num_features,
                         tuple(s // p for s, p in zip(size, patch)), size,
                         mean, std, interpolation=InterpolationMode.BILINEAR)
        self.base_model = model
        self.patch_size = model.patch_size
        self.feature_key = feature_key
        self.layer = layer

    def forward_tokens(self, images):
        if self.layer is not None:
            return self.base_model.get_intermediate_layers(
                images, n=[self.layer], reshape=False,
                return_class_token=False, norm=True,
            )[0]
        return self.base_model.forward_features(images)[self.feature_key]
