"""
Configuration management for the object tracking application.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class AppConfig:
    """
    Configuration class for the object tracking application.
    """
    model_path: Path
    tracker_type: str
    ppm: float
    conf_thres: float
    enable_gpu: bool
    show_paths: bool
    save_video: bool
    export_csv: bool
    filter_class_ids: List[int]
    
    @classmethod
    def create_default(cls) -> AppConfig:
        """Create a default configuration."""
        return cls(
            model_path=Path("yolov9c.pt"),
            tracker_type="deep_sort",
            ppm=8.0,
            conf_thres=0.25,
            enable_gpu=False,
            show_paths=True,
            save_video=False,
            export_csv=False,
            filter_class_ids=[]
        )
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.model_path, Path):
            raise ValueError("model_path must be a Path object")
        
        if self.tracker_type not in ["deep_sort", "bytetrack"]:
            raise ValueError("tracker_type must be 'deep_sort' or 'bytetrack'")
        
        if not 0.0 <= self.conf_thres <= 1.0:
            raise ValueError("conf_thres must be between 0.0 and 1.0")
        
        if self.ppm <= 0:
            raise ValueError("ppm must be positive")
