import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from resnet.resnet import ResNet

class GradCAM:
    """
    Grad-CAM for artifact classifier
    """
    def __init__(self, model: ResNet, target_layer: torch.nn.Module | None = None):
        self.model = model
        
        if target_layer is None:
            target_layer = model.blocks[-1]

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation) # hook onto target layer
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()
        
    def remove_hooks(self):
        """Remove the registered forward / backward hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass

    def generate(self, x: torch.Tensor) -> np.ndarray:
        """
        compute grad cam heatmap for input
        """
        self.model.zero_grad()
        output = self.model(x)

        # gradient of output w.r.t. target layer activations
        output.backward()

        # global average pooling of gradients across spatial dimensions
        weights = self._gradients.mean(dim=(2, 3), keepdim=True) # [1, C, 1, 1]

        # weighted combination of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True) # [1, 1, H', W']
        cam = F.relu(cam)  # keep only positive influences

        # resize to input spatial size and normalise to [0, 1]
        cam = F.interpolate(
            cam,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy() # [H, W]

        # normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_heatmap( # visualisation helper
    spectrogram: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    overlay a Grad-CAM heatmap on a spectrogram image
    """
    # normalize spectrogram to [0, 1] for display
    s_min, s_max = spectrogram.min(), spectrogram.max()
    spec_norm = (spectrogram - s_min) / (s_max - s_min + 1e-8)
    spec_rgb = np.stack([spec_norm] * 3, axis=-1) # [H, W, 3]

    # apply colormap to heatmap
    cmap = cm.get_cmap(colormap)
    heat_rgba = cmap(cam) # [H, W, 4]
    heat_rgb = heat_rgba[..., :3] # [H, W, 3]

    blended = (1 - alpha) * spec_rgb + alpha * heat_rgb
    return np.clip(blended, 0, 1)
