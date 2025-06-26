"""
Visualization module for object tracking results.
"""

import cv2
import numpy as np
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Set

from utils.geometry import compute_color_for_labels, intersect, estimate_speed


class Visualizer:
    """
    Handles visualization of tracking results including bounding boxes,
    trails, speed estimation, and crossing line detection.
    """
    
    def __init__(self, names: List[str]):
        """
        Initialize the visualizer.
        
        Args:
            names: List of class names for labeling
        """
        self.names = names
        self.trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=32))
        self.speed_records: Dict[int, List[float]] = defaultdict(list)
        self.counter: defaultdict[str, int] = defaultdict(int)
    
    def draw(self, frame: np.ndarray, bbox: np.ndarray, track_ids: np.ndarray, 
             cls_ids: np.ndarray, track_line: Tuple[Tuple[int, int], Tuple[int, int]], 
             show_paths: bool, ppm: float) -> np.ndarray:
        """
        Draw tracking results on the frame.
        
        Args:
            frame: Input frame to draw on
            bbox: Bounding boxes array
            track_ids: Track IDs array
            cls_ids: Class IDs array
            track_line: Crossing line coordinates
            show_paths: Whether to show object paths
            ppm: Pixels per meter for speed calculation
            
        Returns:
            Frame with drawn tracking results
        """
        # Draw the tracking line
        cv2.line(frame, track_line[0], track_line[1], (0, 200, 0), 2)
        
        # Clean up inactive tracks
        active_ids = set(track_ids.tolist())
        self._cleanup_inactive_tracks(active_ids)
        
        # Process each tracked object
        for bb, tid, cid in zip(bbox, track_ids, cls_ids):
            self._draw_single_track(frame, bb, tid, cid, track_line, show_paths, ppm)
        
        return frame
    
    def _cleanup_inactive_tracks(self, active_ids: Set[int]) -> None:
        """Remove trails and speed records for inactive tracks."""
        for tid in list(self.trails):
            if tid not in active_ids:
                self.trails.pop(tid, None)
                self.speed_records.pop(tid, None)
    
    def _draw_single_track(self, frame: np.ndarray, bb: np.ndarray, tid: int, cid: int,
                          track_line: Tuple[Tuple[int, int], Tuple[int, int]], 
                          show_paths: bool, ppm: float) -> None:
        """Draw a single tracked object."""
        x1, y1, x2, y2 = map(int, bb)
        center_bottom = (int((x1 + x2) / 2), int(y2))
        
        # Update trail
        self.trails[tid].appendleft(center_bottom)
        
        # Draw path if enabled
        if show_paths and len(self.trails[tid]) > 1:
            self._draw_path(frame, tid, cid)
        
        # Check for line crossing and calculate speed
        if len(self.trails[tid]) >= 2:
            self._check_line_crossing(tid, cid, track_line, ppm)
        
        # Draw bounding box and label
        self._draw_bbox_and_label(frame, bb, tid, cid)
    
    def _draw_path(self, frame: np.ndarray, tid: int, cid: int) -> None:
        """Draw the path trail for a tracked object."""
        trail_points = list(self.trails[tid])
        for p, q in zip(trail_points[:-1], trail_points[1:]):
            cv2.line(frame, p, q, compute_color_for_labels(cid), 2)
    
    def _check_line_crossing(self, tid: int, cid: int, 
                           track_line: Tuple[Tuple[int, int], Tuple[int, int]], 
                           ppm: float) -> None:
        """Check if object crossed the tracking line and calculate speed."""
        if intersect(self.trails[tid][0], self.trails[tid][1], *track_line):
            speed = estimate_speed(self.trails[tid][1], self.trails[tid][0], ppm)
            self.speed_records[tid].append(speed)
            self.counter[self.names[cid]] += 1
    
    def _draw_bbox_and_label(self, frame: np.ndarray, bb: np.ndarray, tid: int, cid: int) -> None:
        """Draw bounding box and label with speed information."""
        x1, y1, x2, y2 = map(int, bb)
        color = compute_color_for_labels(cid)
        
        # Prepare label with speed if available
        label = self.names[cid]
        if self.speed_records[tid]:
            avg_speed = int(sum(self.speed_records[tid]) / len(self.speed_records[tid]))
            label = f"{label} {avg_speed} km/h"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def get_total_crossings(self) -> int:
        """Get total number of objects that crossed the line."""
        return sum(self.counter.values())
    
    def get_active_tracks_count(self) -> int:
        """Get number of currently active tracks."""
        return len(self.trails)
    
    def get_crossing_counts(self) -> Dict[str, int]:
        """Get crossing counts by object class."""
        return dict(self.counter)
