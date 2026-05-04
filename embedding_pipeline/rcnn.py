from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn

Detection = Dict[str, object]
_MODEL_CACHE: Dict[str, object] = {}


def _get_model(model_name: str) -> Any:
    """
    cache and return a model instance
    """
    if model_name == "fasterrcnn_resnet50_fpn": 
        if model_name not in _MODEL_CACHE:
            _MODEL_CACHE["fasterrcnn_resnet50_fpn"] = fasterrcnn_resnet50_fpn(pretrained=True).eval()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return _MODEL_CACHE.get(model_name)


def run_detector(image: torch.Tensor, device: str) -> List[Detection]:
    """
    run a pretrained rcnn detector on an RGB image
    """
    
    model = _get_model("fasterrcnn_resnet50_fpn")
    res = model([image.to(device)])

    boxes = res[0]['boxes'].cpu().detach().numpy()
    confs = res[0]['scores'].cpu().detach().numpy()

    return [
        {"box": [int(x1), int(y1), int(x2), int(y2)], "conf": float(c)}
        for (x1, y1, x2, y2), c in zip(boxes, confs)
    ]


def postprocess_detections(
    detections: List[Detection],
    image_shape: Tuple[int, int, int],
    conf_thresh: float,
    max_detections: int,
) -> List[Detection]:
    """
    sanitize and filter raw detections
    """
    
    H, W = int(image_shape[1]), int(image_shape[2])

    cleaned = []
    for d in detections:
        
        # drop boxes below conf_thresh
        conf = float(d.get("conf", 0.0))
        if conf < conf_thresh:
            continue
        
        # drop non-boxes
        box = d.get("box")
        if len(box) != 4:
            continue

        x1, y1, x2, y2 = box

        # clip to bounds
        x1 = max(0, min(int(x1), W))
        y1 = max(0, min(int(y1), H))
        x2 = max(0, min(int(x2), W))
        y2 = max(0, min(int(y2), H))
  
        # drop degenerate boxes after clipping
        if x2 <= x1 or y2 <= y1:
            continue
  
        cleaned.append({"box": [x1, y1, x2, y2], "conf": conf})

    cleaned.sort(key=lambda dd: dd["conf"], reverse=True) # sort by highest to lowest confidence
    return cleaned[:max_detections]

