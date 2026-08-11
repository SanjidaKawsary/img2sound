import os
import torch
import numpy as np
from encoder.image.imagebind.models.imagebind_model import ModalityType
from encoder.image.imagebind.models import imagebind_model
from encoder.image.imagebind import data


class ImageFeatureExtractor:
    def __init__(self, device="cuda", weights_path=None):
        self.device = device

        if weights_path is None:
            weights_path = os.environ.get(
                "IMAGEBIND_WEIGHTS",
                os.path.join(
                    os.path.dirname(__file__),
                    "imagebind", "models", ".checkpoints", "imagebind_huge.pth",
                ),
            )

        if os.path.exists(weights_path):
            # Build empty model, then load weights explicitly (avoids broken auto-download)
            self.image_encoder = imagebind_model.imagebind_huge(pretrained=False)
            self.image_encoder.load_state_dict(torch.load(weights_path, map_location="cpu"))
            self.image_encoder = self.image_encoder.to(device)
        else:
            # Fall back to the pretrained=True path (which now has a clear error message)
            self.image_encoder = imagebind_model.imagebind_huge(pretrained=True).to(device)

        self.image_encoder.eval()

    def extract_features(self, image_path, save_path):
        image = data.load_and_transform_vision_data([image_path], "cpu")
        image = {ModalityType.VISION: image.to(self.device, non_blocking=True).squeeze(1)}
        with torch.no_grad():
            image = self.image_encoder(image)[ModalityType.VISION]
        z = image.detach().cpu().numpy().squeeze()
        np.save(save_path, z)
