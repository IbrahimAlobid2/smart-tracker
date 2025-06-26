# AI Workshop
> **Created and maintained by : Ibrahim Alobaid**  
> [GitHub](https://github.com/IbrahimAlobid2) | [Linkedin](https://www.linkedin.com/in/ibrahimalobaid44/) .

> AI Engineer | Computer Vision & Deep Learning Specialist |  Tech Enthusiast. | Co-Founder @ [Aleppo Dev Community](https://aleppo.dev/) |

# Object Tracking System 

A comprehensive real-time object tracking system built with Streamlit, featuring YOLO-based detection, multiple tracking algorithms, and Arabic language support.

##  Features

- **Real-time Object Detection**: YOLO-based detection with multiple model support (YOLOv5, YOLOv8, YOLOv9)
- **Multi-Object Tracking**: DeepSORT and ByteTrack algorithms
- **Speed Estimation**: Calculate object speeds with pixel-per-meter calibration
- **Line Crossing Detection**: Count objects crossing predefined lines
- **Arabic Interface**: Full RTL support with Arabic text
- **Data Export**: CSV export of tracking data and processed video saving
- **Interactive UI**: Streamlit-based web interface with real-time metrics
- **Class Filtering**: Filter detection by specific object classes
- **GPU Support**: Optional CUDA acceleration for faster processing

##  System Architecture

The system follows a modular architecture with clear separation of concerns:

```
├── config/           # Configuration management
│   └── app_config.py
├── core/            # Core processing modules
│   ├── detector.py     # YOLO-based object detection
│   ├── tracker.py      # Multi-object tracking
│   └── visualizer.py   # Result visualization
├── services/        # Business logic services
│   └── video_processor.py
├── utils/           # Utility functions
│   ├── constants.py
│   └── geometry.py
├── deep_sort_pytorch/  # DeepSORT implementation
└── app.py          # Main Streamlit application
```

##  Quick Start

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (optional, for acceleration)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd object-tracking-system
   ```

2. **Install dependencies** (Option 1 - Recommended for development):
   ```bash
   # Install basic packages for the simplified version
   pip install streamlit pandas numpy
   ```

3. **Install full dependencies** (Option 2 - For complete functionality):
   ```bash
   # Install all required packages
   pip install opencv-python torch torchvision ultralytics pandas streamlit streamlit-drawable-canvas
   ```

4. **Setup DeepSORT** (Optional - for advanced tracking):
   ```bash
   # Extract the provided deep_sort_pytorch.zip to the project root
   # The deep_sort_pytorch directory should contain the necessary files
   ```

### Running the Application

**Simplified Version** (Works with basic dependencies):
```bash
streamlit run app_simple.py --server.port 5000
```

**Full Version** (Requires all dependencies):
```bash
streamlit run app.py --server.port 5000
```

**Access the application**:
Open your browser and navigate to `http://localhost:5000`

##  Usage Guide

### Basic Workflow

1. **Configure Settings** :
   - **Model Selection**: Choose from available YOLO models
   - **Confidence Threshold**: Set detection confidence (0.0-1.0)
   - **Pixels Per Meter (PPM)**: Calibrate for speed calculation
   - **Tracker Type**: Select DeepSORT or ByteTrack
   - **Processing Options**: Enable GPU, show paths, save video, export CSV

2. **Upload Video**:
   - Use the file uploader to select your video file
   - Supported formats: MP4, AVI, MOV
   - Or use the demo video if available

3. **Class Filtering** (Optional):
   - Enable "Use custom classes"
   - Select specific object classes to track

4. **Start Processing**:
   - Click the " Start" button
   - Monitor real-time metrics: FPS, Active IDs, Total Crossed

### Configuration Options

#### Model Options
- `yolov8_kitti`: YOLOv8 trained on KITTI dataset
- `yolov9_kitti`: YOLOv9 trained on KITTI dataset  
- `yolov5kitti`: YOLOv5 trained on KITTI dataset


#### Tracker Options
- **DeepSORT**: Robust tracking with appearance features
- **ByteTrack**: High-performance tracking (requires additional dependencies)

#### Export Options
- **Save Video**: Export processed video with tracking overlays
- **CSV Export**: Export tracking data with timestamps and coordinates

##  Configuration

### Application Configuration (`config/app_config.py`)

The `AppConfig` class manages all application settings:

```python
@dataclass
class AppConfig:
    model_path: Path          # Path to YOLO model
    tracker_type: str         # 'deep_sort' or 'bytetrack'
    ppm: float               # Pixels per meter
    conf_thres: float        # Confidence threshold
    enable_gpu: bool         # GPU acceleration
    show_paths: bool         # Show object trails
    save_video: bool         # Save processed video
    export_csv: bool         # Export tracking data
    filter_class_ids: List[int]  # Class filtering
```

### Model Files

Place your YOLO model files in the `models/` directory:
- `models/yolov8_kitti.pt`
- `models/yolov9_kitti.pt`
- `models/yolov5kitti.pt`


### DeepSORT Configuration

DeepSORT settings are configured in `deep_sort_pytorch/configs/deep_sort.yaml`:

```yaml
DEEPSORT:
  REID_CKPT: "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
  MAX_DIST: 0.2
  MIN_CONFIDENCE: 0.3
  NMS_MAX_OVERLAP: 1.0
  MAX_IOU_DISTANCE: 0.7
  MAX_AGE: 70
  N_INIT: 3
  NN_BUDGET: 100
```

##  Output Data

### CSV Export Format

When CSV export is enabled, the system generates `tracking_data.csv` with:

```csv
time,track_id,class,x1,y1,x2,y2
1640995200.123,1,car,100,200,300,400
1640995200.156,2,person,150,180,200,350
```

### Video Output

Processed videos are saved as `processed.mp4` with:
- Bounding boxes around detected objects
- Track IDs and class labels
- Speed information (when available)
- Object trails/paths (if enabled)
- Crossing line visualization

##  Development

### Project Structure

- **`core/`**: Core processing modules
  - `detector.py`: YOLO-based object detection
  - `tracker.py`: Multi-object tracking algorithms
  - `visualizer.py`: Drawing and visualization

- **`services/`**: High-level business logic
  - `video_processor.py`: Main video processing pipeline

- **`utils/`**: Utility functions
  - `constants.py`: Application constants
  - `geometry.py`: Geometric calculations

- **`config/`**: Configuration management
  - `app_config.py`: Application configuration class

### Adding New Models

1. Place the model file in the `models/` directory
2. Update `MODEL_MAP` in `utils/constants.py`:
   ```python
   MODEL_MAP = {
       "your_model": "models/your_model.pt",
       # ... existing models
   }
   ```

### Adding New Trackers

1. Implement the tracker in `core/tracker.py`
2. Add the new tracker type to the `TrackerWrapper` class
3. Update the UI options in `app.py`

##  Troubleshooting

### Common Issues

1. **Missing Model Files**:
   - Ensure YOLO model files exist in the `models/` directory

2. **DeepSORT Checkpoint Missing**:
   - Verify `deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7` exists
   - Download from the DeepSORT repository if needed

3. **GPU Issues**:
   - Check CUDA installation with `torch.cuda.is_available()`
   - Disable GPU in settings if having issues

4. **Video Upload Problems**:
   - Ensure video format is supported (MP4, AVI, MOV)
   - Check file size limitations

### Performance Optimization

- **GPU Acceleration**: Enable for faster processing on CUDA-compatible hardware
- **Lower Confidence**: Reduce false positives but may miss detections
- **Frame Skipping**: Process every nth frame for faster processing
- **Model Selection**: Smaller models process faster but may be less accurate

##  Requirements

### Python Dependencies

```
opencv-python>=4.5.0
torch>=1.9.0
torchvision>=0.10.0
ultralytics>=8.0.0
pandas>=1.3.0
streamlit>=1.25.0
streamlit-drawable-canvas>=0.9.0
numpy>=1.21.0
```

### System Requirements

- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: 2GB for models and dependencies
- **GPU**: Optional CUDA-compatible GPU for acceleration

##  Deployment

### Replit Deployment

The application is configured for Replit deployment:

1. **Server Configuration** (`.streamlit/config.toml`):
   ```toml
   [server]
   headless = true
   address = "0.0.0.0"
   port = 5000
   ```

2. **Workflow Configuration**:
   ```bash
   streamlit run app.py --server.port 5000
   ```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["streamlit", "run", "app.py", "--server.port", "5000", "--server.address", "0.0.0.0"]
```

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


##  Acknowledgments

- **Ultralytics**: YOLO implementation
- **DeepSORT**: Multi-object tracking algorithm
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision library

a new issue with detailed information

---

**Note**: This system is designed for research and educational purposes. For production use, consider additional optimizations and testing.
