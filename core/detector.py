"""
Object detection module using YOLO models.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple
from ultralytics import YOLO


class Detector:
    """
    YOLO-based object detector for tracking applications.
    """
    
    def __init__(self, model_path: Path, device: str):
        """
        Initialize the detector with a YOLO model.
        
        Args:
            model_path: Path to the YOLO model file
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model = YOLO(str(model_path)).to(device)
        self.device = device
    
    def __call__(self, frame: np.ndarray, conf: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input image/frame
            conf: Confidence threshold for detections
            
        Returns:
            Tuple of (bounding_boxes, confidences, class_ids)
        """
        try:
            results = self.model.predict(frame, conf=conf, device=self.device)
            
            # Handle empty results
            if not results or results[0].boxes.xyxy is None:
                return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            return boxes, confs, classes
            
        except Exception as e:
            print(f"Detection error: {e}")
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
