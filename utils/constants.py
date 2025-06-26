"""
Constants used throughout the application.
"""

DEFAULT_NAMES_COCO = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
]

DEFAULT_NAMES_KITTI = [
    'Car', 
    'Van', 
    'Truck', 
    'Pedestrian',
    'Person_sitting',
    'Cyclist',
    'Tram', 
    'Misc'
]


# Default model mappings
MODEL_MAP = {
    "yolov8_kitti": "models/yolov8_kitti.pt",
    "yolov9_kitti": "models/yolov9_kitti.pt",
    "yolov5kitti": "models/yolov5kitti.pt",
    "yolov9": "models/yolov9c.pt",
}

# Default tracking line coordinates
DEFAULT_LINE_COORDS = ((200, 500), (1050, 500))

# Video processing constants
DEFAULT_FPS = 25
VIDEO_OUTPUT_CODEC = "mp4v"
