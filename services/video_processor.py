"""
Video processing service for object tracking.
"""

import cv2
import numpy as np
import pandas as pd
import time
import tempfile
from typing import Optional, List, Dict, Any, Tuple
import streamlit as st

from config.app_config import AppConfig
from core.detector import Detector
from core.tracker import TrackerWrapper
from core.visualizer import Visualizer
from utils.constants import DEFAULT_NAMES, DEFAULT_FPS, VIDEO_OUTPUT_CODEC


class VideoProcessor:
    """
    Handles video processing for object tracking including detection,
    tracking, visualization, and data export.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize the video processor.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.device = self._get_device()
        self.detector = Detector(config.model_path, self.device)
        self.tracker = TrackerWrapper(config.tracker_type, config.enable_gpu)
        self.visualizer = Visualizer(DEFAULT_NAMES)
        
    def _get_device(self) -> str:
        """Determine the appropriate device for processing."""
        import torch
        if self.config.enable_gpu and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def process_video(self, video_source: str, track_line: Tuple[Tuple[int, int], Tuple[int, int]], 
                     stframe, kpi1, kpi2, kpi3) -> Optional[List[Dict[str, Any]]]:
        """
        Process video with object tracking.
        
        Args:
            video_source: Path to video file or camera index
            track_line: Coordinates of the tracking line
            stframe: Streamlit frame placeholder
            kpi1, kpi2, kpi3: Streamlit metric placeholders
            
        Returns:
            List of tracking data records if CSV export is enabled
        """
        cap = self._initialize_video_capture(video_source)
        if cap is None:
            st.error("❌ فشل في فتح مصدر الفيديو")
            return None
        
        out_writer = self._initialize_video_writer(cap) if self.config.save_video else None
        csv_rows = []
        prev_time = 0
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                processed_frame, tracking_data = self._process_frame(frame, track_line)
                
                # Update metrics and display
                current_time = time.time()
                fps = self._calculate_fps(current_time, prev_time)
                prev_time = current_time
                
                self._update_kpis(kpi1, kpi2, kpi3, fps)
                stframe.image(processed_frame, channels="BGR", use_container_width=True)
                
                # Save processed frame if enabled
                if out_writer is not None:
                    out_writer.write(processed_frame)
                
                # Collect CSV data if enabled
                if self.config.export_csv and tracking_data:
                    csv_rows.extend(self._format_csv_data(tracking_data, current_time))
                    
        except Exception as e:
            st.error(f"❌ خطأ في معالجة الفيديو: {str(e)}")
        finally:
            self._cleanup_resources(cap, out_writer)
        
        # Handle CSV export
        if self.config.export_csv and csv_rows:
            self._export_csv_data(csv_rows)
        
        return csv_rows if csv_rows else None
    
    def _initialize_video_capture(self, video_source: str) -> Optional[cv2.VideoCapture]:
        """Initialize video capture from source."""
        try:
            source = 0 if video_source == "0" else video_source
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                return None
            return cap
        except Exception as e:
            st.error(f"❌ خطأ في تهيئة مصدر الفيديو: {str(e)}")
            return None
    
    def _initialize_video_writer(self, cap: cv2.VideoCapture) -> Optional[cv2.VideoWriter]:
        """Initialize video writer for saving processed video."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_OUTPUT_CODEC)
            fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            return cv2.VideoWriter("processed.mp4", fourcc, fps, (width, height))
        except Exception as e:
            st.warning(f"⚠️ تحذير: فشل في تهيئة كاتب الفيديو: {str(e)}")
            return None
    
    def _process_frame(self, frame: np.ndarray, track_line: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """Process a single frame for detection and tracking."""
        # Object detection
        boxes, confs, clss = self.detector(frame, conf=self.config.conf_thres)
        
        # Apply class filtering if specified
        if self.config.filter_class_ids:
            mask = np.isin(clss, np.array(self.config.filter_class_ids))
            boxes, confs, clss = boxes[mask], confs[mask], clss[mask]
        
        # Skip tracking if no detections
        if boxes.size == 0:
            return frame, None
        
        # Convert to tracking format (xywh)
        xywhs = self._convert_to_xywh(boxes)
        
        # Object tracking
        bbox, track_ids, cls_ids = self.tracker.update(xywhs, confs, clss, frame)
        
        # Visualization
        processed_frame = self.visualizer.draw(
            frame, bbox, track_ids, cls_ids, track_line, 
            self.config.show_paths, self.config.ppm
        )
        
        # Prepare tracking data for CSV export
        tracking_data = self._prepare_tracking_data(bbox, track_ids, cls_ids) if bbox.size > 0 else None
        
        return processed_frame, tracking_data
    
    def _convert_to_xywh(self, boxes: np.ndarray) -> np.ndarray:
        """Convert bounding boxes from xyxy to xywh format."""
        return np.column_stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,  # center_x
            (boxes[:, 1] + boxes[:, 3]) / 2,  # center_y
            boxes[:, 2] - boxes[:, 0],        # width
            boxes[:, 3] - boxes[:, 1],        # height
        ])
    
    def _prepare_tracking_data(self, bbox: np.ndarray, track_ids: np.ndarray, 
                              cls_ids: np.ndarray) -> List[Dict[str, Any]]:
        """Prepare tracking data for CSV export."""
        tracking_data = []
        for bb, tid, cid in zip(bbox, track_ids, cls_ids):
            tracking_data.append({
                "track_id": int(tid),
                "class": DEFAULT_NAMES[int(cid)],
                "x1": float(bb[0]), "y1": float(bb[1]),
                "x2": float(bb[2]), "y2": float(bb[3])
            })
        return tracking_data
    
    def _format_csv_data(self, tracking_data: List[Dict[str, Any]], 
                        current_time: float) -> List[Dict[str, Any]]:
        """Format tracking data with timestamp for CSV export."""
        csv_data = []
        for data in tracking_data:
            data["time"] = current_time
            csv_data.append(data)
        return csv_data
    
    def _calculate_fps(self, current_time: float, prev_time: float) -> float:
        """Calculate current FPS."""
        if current_time > prev_time:
            return 1.0 / (current_time - prev_time)
        return 0.0
    
    def _update_kpis(self, kpi1, kpi2, kpi3, fps: float) -> None:
        """Update KPI metrics in the UI."""
        kpi1.metric(label="FPS", value=f"{fps:.1f}")
        kpi2.metric(label="Active IDs", value=f"{self.visualizer.get_active_tracks_count()}")
        kpi3.metric(label="Total Crossed", value=self.visualizer.get_total_crossings())
    
    def _export_csv_data(self, csv_rows: List[Dict[str, Any]]) -> None:
        """Export tracking data to CSV file."""
        try:
            df = pd.DataFrame(csv_rows)
            csv_filename = "tracking_data.csv"
            df.to_csv(csv_filename, index=False)
            
            st.success("👉 تم حفظ بيانات التتبع فى tracking_data.csv")
            
            # Provide download button
            with open(csv_filename, "rb") as f:
                st.download_button(
                    "Download CSV", 
                    f, 
                    csv_filename, 
                    "text/csv"
                )
        except Exception as e:
            st.error(f"❌ خطأ في تصدير CSV: {str(e)}")
    
    def _cleanup_resources(self, cap: Optional[cv2.VideoCapture], 
                          out_writer: Optional[cv2.VideoWriter]) -> None:
        """Clean up video capture and writer resources."""
        if cap is not None:
            cap.release()
        if out_writer is not None:
            out_writer.release()
