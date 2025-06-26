"""
Simple tracking implementation as fallback when DeepSORT is unavailable.
"""

import numpy as np
from typing import Tuple, Dict, List


class SimpleTracker:
    """
    Simple centroid-based tracker for when DeepSORT is unavailable.
    """
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 50):
        """
        Initialize the simple tracker.
        
        Args:
            max_disappeared: Maximum frames an object can be missing before removal
            max_distance: Maximum distance for object association
        """
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid: Tuple[float, float], class_id: int):
        """Register a new object."""
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'class_id': class_id
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id: int):
        """Deregister an object."""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def _compute_distance_matrix(self, centroids1, centroids2):
        """Compute distance matrix between two sets of centroids."""
        centroids1 = np.array(centroids1)
        centroids2 = np.array(centroids2)
        
        # Calculate Euclidean distance matrix
        diff = centroids1[:, np.newaxis, :] - centroids2[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        
        return distances
    
    def update(self, rects: np.ndarray, class_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update tracker with new detections.
        
        Args:
            rects: Bounding boxes in xyxy format
            class_ids: Class IDs for each detection
            
        Returns:
            Tuple of (bounding_boxes, track_ids, class_ids)
        """
        if len(rects) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0, dtype=int)
        
        # Compute centroids for new detections
        input_centroids = []
        for (x1, y1, x2, y2) in rects:
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids.append((cx, cy))
        
        if len(self.objects) == 0:
            # Register all detections as new objects
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, class_ids[i])
        else:
            # Get existing object centroids
            object_centroids = [obj['centroid'] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())
            
            # Compute distances between existing and new centroids
            D = self._compute_distance_matrix(object_centroids, input_centroids)
            
            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            # Keep track of used row and column indices
            used_row_indices = set()
            used_col_indices = set()
            
            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id]['centroid'] = input_centroids[col]
                self.objects[object_id]['class_id'] = class_ids[col]
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            if D.shape[0] >= D.shape[1]:
                # More existing objects than detections
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # More detections than existing objects
                for col in unused_col_indices:
                    self.register(input_centroids[col], class_ids[col])
        
        # Prepare output
        if len(self.objects) == 0:
            return np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0, dtype=int)
        
        # Match objects to input rectangles
        output_rects = []
        output_ids = []
        output_classes = []
        
        for i, (object_id, obj_data) in enumerate(self.objects.items()):
            # Find closest input rectangle to this object
            obj_centroid = obj_data['centroid']
            min_dist = float('inf')
            best_rect_idx = -1
            
            for j, input_centroid in enumerate(input_centroids):
                dist = np.linalg.norm(np.array(obj_centroid) - np.array(input_centroid))
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    best_rect_idx = j
            
            if best_rect_idx != -1:
                output_rects.append(rects[best_rect_idx])
                output_ids.append(object_id)
                output_classes.append(obj_data['class_id'])
        
        if len(output_rects) == 0:
            return np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0, dtype=int)
        
        return (np.array(output_rects), 
                np.array(output_ids, dtype=int), 
                np.array(output_classes, dtype=int))