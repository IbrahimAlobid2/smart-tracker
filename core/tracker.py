"""
Object tracking module supporting multiple tracking algorithms.
"""

import numpy as np
from typing import Tuple

# Import fallback tracker
from core.simple_tracker import SimpleTracker

# Import tracking libraries with fallbacks
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from deep_sort_pytorch.utils.parser import get_config
    from deep_sort_pytorch.deep_sort import DeepSort
    _DEEPSORT_AVAILABLE = True
except ImportError:
    _DEEPSORT_AVAILABLE = False

try:
    from yolox.tracker.byte_tracker import BYTETracker  # type: ignore
    _BYTE_AVAILABLE = True
except ImportError:
    _BYTE_AVAILABLE = False


class TrackerWrapper:
    """
    Wrapper class for different tracking algorithms (DeepSORT, ByteTrack).
    """
    
    def __init__(self, tracker_type: str, enable_gpu: bool):
        """
        Initialize the tracker wrapper.
        
        Args:
            tracker_type: Type of tracker ('deep_sort' or 'bytetrack')
            enable_gpu: Whether to use GPU acceleration
        """
        self.type = tracker_type
        self.tracker = self._init_tracker(tracker_type, enable_gpu)
    
    @staticmethod
    def _init_tracker(tracker_type: str, enable_gpu: bool):
        """
        Initialize the appropriate tracker based on type.
        
        Args:
            tracker_type: Type of tracker to initialize
            enable_gpu: Whether to enable GPU acceleration
            
        Returns:
            Initialized tracker instance
        """
        if tracker_type == "bytetrack" and _BYTE_AVAILABLE:
            return BYTETracker()
        
        if tracker_type == "deep_sort" and _DEEPSORT_AVAILABLE and _TORCH_AVAILABLE:
            try:
                import torch
                cfg = get_config()
                cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
                return DeepSort(
                    cfg.DEEPSORT.REID_CKPT, 
                    use_cuda=enable_gpu and torch.cuda.is_available()
                )
            except Exception:
                pass
        
        # Fallback to simple tracker
        return SimpleTracker()
    
    def update(self, xywhs: np.ndarray, confs: np.ndarray, 
               clss: np.ndarray, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update tracker with new detections.
        
        Args:
            xywhs: Bounding boxes in xywh format
            confs: Detection confidences
            clss: Class IDs
            frame: Current frame
            
        Returns:
            Tuple of (bounding_boxes, track_ids, class_ids)
        """
        try:
            if self.type == "bytetrack" and _BYTE_AVAILABLE:
                return self._update_bytetrack(xywhs, confs, clss, frame)
            elif self.type == "deep_sort" and _DEEPSORT_AVAILABLE:
                return self._update_deepsort(xywhs, confs, clss, frame)
            else:
                return self._update_simple(xywhs, confs, clss, frame)
        except Exception as e:
            print(f"Tracker update error: {e}")
            return np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0, dtype=int)
    
    def _update_bytetrack(self, xywhs: np.ndarray, confs: np.ndarray, 
                         clss: np.ndarray, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update using ByteTrack algorithm."""
        online_targets = self.tracker.update(
            np.concatenate([xywhs, confs[:, None]], axis=1), clss, frame.shape
        )
        
        if len(online_targets) == 0:
            return np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0, dtype=int)
        
        bbox = np.array([t.tlbr for t in online_targets])
        track_ids = np.array([t.track_id for t in online_targets])
        cls_ids = np.array([t.cls for t in online_targets])
        
        return bbox, track_ids, cls_ids
    
    def _update_deepsort(self, xywhs: np.ndarray, confs: np.ndarray, 
                        clss: np.ndarray, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update using DeepSORT algorithm."""
        if not _TORCH_AVAILABLE:
            return np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0, dtype=int)
        
        import torch
        outputs = self.tracker.update(
            torch.Tensor(xywhs), torch.Tensor(confs), clss, frame
        )
        
        if len(outputs) == 0:
            return np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0, dtype=int)
        
        bbox = outputs[:, :4]
        track_ids = outputs[:, -2]
        cls_ids = outputs[:, -1]
        
        return bbox, track_ids, cls_ids
    
    def _update_simple(self, xywhs: np.ndarray, confs: np.ndarray, 
                      clss: np.ndarray, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update using simple tracker algorithm."""
        if len(xywhs) == 0:
            return self.tracker.update(np.empty((0, 4)), np.empty(0, dtype=int))
        
        # Convert xywh to xyxy format for simple tracker
        xyxy_boxes = np.zeros((len(xywhs), 4))
        xyxy_boxes[:, 0] = xywhs[:, 0] - xywhs[:, 2] / 2  # x1
        xyxy_boxes[:, 1] = xywhs[:, 1] - xywhs[:, 3] / 2  # y1
        xyxy_boxes[:, 2] = xywhs[:, 0] + xywhs[:, 2] / 2  # x2
        xyxy_boxes[:, 3] = xywhs[:, 1] + xywhs[:, 3] / 2  # y2
        
        return self.tracker.update(xyxy_boxes, clss.astype(int))

    @staticmethod
    def is_bytetrack_available() -> bool:
        """Check if ByteTrack is available."""
        return _BYTE_AVAILABLE
