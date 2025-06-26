"""
Geometric utility functions for object tracking and speed estimation.
"""

import numpy as np
from typing import Tuple


def compute_color_for_labels(label: int) -> Tuple[int, int, int]:
    """
    Generate a consistent color for a given label ID.
    
    Args:
        label: Integer label ID
        
    Returns:
        RGB color tuple
    """
    np.random.seed(label)
    return tuple(map(int, np.random.randint(0, 255, size=3)))


def intersect(A: Tuple[int, int], B: Tuple[int, int], 
              C: Tuple[int, int], D: Tuple[int, int]) -> bool:
    """
    Check if line segment AB intersects with line segment CD.
    
    Args:
        A, B: Points defining first line segment
        C, D: Points defining second line segment
        
    Returns:
        True if segments intersect, False otherwise
    """
    def ccw(P: Tuple[int, int], Q: Tuple[int, int], R: Tuple[int, int]) -> bool:
        return (R[1] - P[1]) * (Q[0] - P[0]) > (Q[1] - P[1]) * (R[0] - P[0])
    
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def estimate_speed(p1: Tuple[int, int], p2: Tuple[int, int], 
                  ppm: float, time_factor: float = 3.6 * 15) -> float:
    """
    Estimate speed based on pixel movement and pixels per meter.
    
    Args:
        p1: Previous position (x, y)
        p2: Current position (x, y)
        ppm: Pixels per meter conversion factor
        time_factor: Time conversion factor (default: 3.6 * 15 for km/h)
        
    Returns:
        Estimated speed in km/h
    """
    d_pixels = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
    d_meters = d_pixels / ppm
    return d_meters * time_factor
