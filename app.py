"""
Main Streamlit application for object tracking system.
"""

import streamlit as st
import tempfile
from pathlib import Path
from typing import List, Optional

from config.app_config import AppConfig
from services.video_processor import VideoProcessor
from utils.constants import DEFAULT_NAMES, MODEL_MAP, DEFAULT_LINE_COORDS
from core.tracker import TrackerWrapper


def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="منظومة تتبع الأهداف",
        page_icon="📍",
        layout="wide"
    )


def create_sidebar() -> AppConfig:
    """
    Create sidebar with configuration options.
    
    Returns:
        Configured AppConfig object
    """
    st.sidebar.header("⚙️ الإعدادات")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Model", 
        list(MODEL_MAP.keys()), 
        index=0
    )
    
    # Detection parameters
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25, 0.05)
    ppm = st.sidebar.number_input("Pixels per meter (PPM)", 1.0, 100.0, 8.0)
    
    # Tracker selection
    available_trackers = ["deep_sort"]
    if TrackerWrapper.is_bytetrack_available():
        available_trackers.append("bytetrack")
    
    tracker_type = st.sidebar.selectbox(
        "Tracker", 
        available_trackers, 
        index=1 if "bytetrack" in available_trackers else 0
    )
    
    # Processing options
    enable_gpu = st.sidebar.checkbox("Enable GPU", value=False)
    show_paths = st.sidebar.checkbox("Show Object Paths", value=True)
    save_video = st.sidebar.checkbox("Save processed video")
    export_csv = st.sidebar.checkbox("Export CSV log")
    
    # Class filtering
    filter_class_ids = setup_class_filtering()
    
    return AppConfig(
        model_path=Path(MODEL_MAP[model_choice]),
        tracker_type=tracker_type,
        ppm=ppm,
        conf_thres=confidence,
        enable_gpu=enable_gpu,
        show_paths=show_paths,
        save_video=save_video,
        export_csv=export_csv,
        filter_class_ids=filter_class_ids
    )


def setup_class_filtering() -> List[int]:
    """
    Setup class filtering options in sidebar.
    
    Returns:
        List of selected class IDs
    """
    custom_classes = st.sidebar.checkbox("Use custom classes")
    class_ids = []
    
    if custom_classes:
        selected_classes = st.sidebar.multiselect(
            "Select classes", 
            DEFAULT_NAMES, 
            default=["car"]
        )
        class_ids = [DEFAULT_NAMES.index(cls) for cls in selected_classes]
    
    return class_ids


def setup_video_source() -> Optional[str]:
    """
    Setup video source selection in sidebar.
    
    Returns:
        Path to video source or None if no source selected
    """
    video_file = st.sidebar.file_uploader(
        "Upload video", 
        type=["mp4", "avi", "mov"]
    )
    
    demo_path = "test.mp4"
    
    if video_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            return tmp_file.name
    else:
        # Show demo video preview if available
        try:
            with open(demo_path, "rb") as demo_file:
                st.sidebar.video(demo_file.read())
            return demo_path
        except FileNotFoundError:
            st.sidebar.warning("⚠️ لا يوجد فيديو تجريبي متاح")
            return None


def setup_main_interface():
    """Setup main interface components."""
    st.title("📍 منظومة تتبع الأهداف")
    
    # Create placeholders for video and metrics
    stframe = st.empty()
    kpi1, kpi2, kpi3 = st.columns(3)
    
    return stframe, kpi1, kpi2, kpi3


def validate_configuration(config: AppConfig) -> bool:
    """
    Validate application configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if configuration is valid
    """
    try:
        config.validate()
        
        # Check if model file exists
        if not config.model_path.exists() and config.model_path.name != "yolov9c.pt":
            st.error(f"❌ ملف النموذج غير موجود: {config.model_path}")
            return False
        
        return True
    except ValueError as e:
        st.error(f"❌ خطأ في التكوين: {str(e)}")
        return False


def main():
    """Main application function."""
    setup_page_config()
    
    # Setup UI components
    config = create_sidebar()
    video_source = setup_video_source()
    stframe, kpi1, kpi2, kpi3 = setup_main_interface()
    
    # Processing button and logic
    if st.sidebar.button("🚀 Start"):
        if not validate_configuration(config):
            return
        
        if video_source is None:
            st.error("❌ يرجى تحديد مصدر الفيديو")
            return
        
        try:
            # Initialize video processor
            processor = VideoProcessor(config)
            
            # Process video
            with st.spinner("🔄 جاري معالجة الفيديو..."):
                csv_data = processor.process_video(
                    video_source, 
                    DEFAULT_LINE_COORDS, 
                    stframe, 
                    kpi1, 
                    kpi2, 
                    kpi3
                )
            
            st.success("✅ المعالجة انتهت")
            
            # Display summary if CSV data is available
            if csv_data:
                st.info(f"📊 تم معالجة {len(csv_data)} نقطة بيانات للتتبع")
                
        except Exception as e:
            st.error(f"❌ خطأ أثناء المعالجة: {str(e)}")
    
    # Add usage instructions
    with st.expander("📋 تعليمات الاستخدام"):
        st.markdown("""
        ### كيفية استخدام النظام:
        
        1. **اختر النموذج**: حدد نموذج YOLO المناسب من القائمة
        2. **اضبط المعاملات**: 
           - Confidence: حد الثقة للكشف (0.0-1.0)
           - PPM: عدد البكسل لكل متر لحساب السرعة
        3. **اختر خوارزمية التتبع**: DeepSORT أو ByteTrack (إذا كان متاحاً)
        4. **حمّل الفيديو**: أو استخدم الفيديو التجريبي
        5. **ابدأ المعالجة**: اضغط على زر "Start"
        
        ### الميزات:
        - تتبع الأهداف في الوقت الفعلي
        - تقدير السرعة للأهداف المتحركة
        - عد الأهداف التي تعبر الخط المحدد
        - حفظ الفيديو المعالج
        - تصدير بيانات التتبع إلى CSV
        """)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    except Exception as e:
        st.error(f"❌ خطأ في التطبيق: {str(e)}")
