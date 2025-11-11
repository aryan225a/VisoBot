"""
VisoBot - AI-Powered Object Detection and Assistant
Optimized version with Gemini API and real-time webcam detection
"""

import cv2
import streamlit as st
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import google.genai as genai
import os
from dotenv import load_dotenv
import time
import numpy as np

load_dotenv()

# ---------- Non-invasive helpers added to reduce lag and jitter ----------
import threading
import time

class VideoGet:
    """Threaded capture helper. Non-destructive: behaves like cv2.VideoCapture for read()."""
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        try:
            # try to reduce internal driver buffer where supported
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        t = threading.Thread(target=self._update, daemon=True)
        t.start()

    def _update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
            # tiny sleep to avoid busy loop if the camera is weird
            time.sleep(0.001)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.stopped = True
        try:
            self.cap.release()
        except Exception:
            pass

def video_read(cap_like):
    """
    Non-invasive wrapper for reading frames.
    If `cap_like` is a VideoGet instance, use its read(); otherwise call cap_like.read().
    This lets us replace all `ret, frame = cap.read()` with `ret, frame = video_read(cap)` safely.
    """
    try:
        # VideoGet instance
        if hasattr(cap_like, 'read') and getattr(cap_like, '_update', None) is not None:
            return cap_like.read()
    except Exception:
        pass
    # fallback to original behaviour (cv2.VideoCapture or similar)
    return cap_like.read()
# -------------------------------------------------------------------------


# Page configuration
st.set_page_config(
    page_title="VisoBot - AI Object Detection",
    page_icon="🔷",
    layout="wide"
)

# Initialize session state
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False
if 'tracked_objects' not in st.session_state:
    st.session_state.tracked_objects = set()

# Load models and APIs
@st.cache_resource
def load_models():
    """Load YOLO and DeepSORT models"""
    try:
        model = YOLO("yolov8n.pt")
        # Optimized DeepSORT settings for stability
        tracker = DeepSort(
            max_age=50,           # Keep tracks longer
            n_init=5,             # More frames before confirmation
            max_iou_distance=0.7, # More lenient matching
            max_cosine_distance=0.3,
            nn_budget=100
        )
        return model, tracker
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Initialize Gemini API
def init_gemini():
    """Initialize Google Gemini API"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.warning("⚠️ Gemini API key not found. AI assistant features disabled.")
        return None
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing Gemini: {e}")
        return None

# Gemini AI assistant with caching
@st.cache_data(ttl=3600, show_spinner=False)
def chat_with_gemini(_object_name, question, _client):
    """Get AI response about detected object"""
    if not _client:
        return "AI assistant not available. Please check your API key."
    
    try:
        prompt = f"""You are an intelligent assistant helping users learn about objects.

Object detected: {_object_name}
User question: {question}

Provide a clear, concise, and informative answer in 2-3 sentences."""
        
        response = _client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error getting AI response: {str(e)}"

# Main app
def main():
    # Header
    st.title("🔷 VisoBot - AI Object Detection & Assistant")
    st.markdown("*Real-time object detection with intelligent Q&A powered by YOLOv8 & Gemini AI*")
    
    # Load models
    model, tracker = load_models()
    gemini_model = init_gemini()
    
    if model is None:
        st.error("Failed to load detection models. Please check installation.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Detection controls
        st.subheader("📹 Camera Control")
        start_detection = st.button("🟢 Start Detection", use_container_width=True)
        stop_detection = st.button("🔴 Stop Detection", use_container_width=True)
        
        if start_detection:
            st.session_state.detection_active = True
        if stop_detection:
            st.session_state.detection_active = False
        
        st.divider()
        
        # Detection settings
        st.subheader("🎯 Detection Settings")
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.65, 0.05)
        process_every_n_frames = st.slider("Process Every N Frames", 1, 5, 2)
        
        st.divider()
        
        # AI Assistant
        st.subheader("🤖 AI Assistant")
        object_query = st.text_input("Object to learn about:", placeholder="e.g., apple, laptop")
        user_question = st.text_area("Ask a question:", placeholder="e.g., What are its nutritional benefits?")
        
        ask_button = st.button("🔍 Get Information", use_container_width=True)
        
        st.divider()
        
        # Detected objects
        st.subheader("📊 Detected Objects")
        if st.session_state.tracked_objects:
            for obj in sorted(st.session_state.tracked_objects):
                st.markdown(f"• {obj}")
        else:
            st.info("No objects detected yet")
        
        if st.button("Clear List", use_container_width=True):
            st.session_state.tracked_objects.clear()
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Detection Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col2:
        st.subheader("📖 Information Panel")
        info_placeholder = st.empty()
    
    # Handle AI queries
    if ask_button and object_query:
        with info_placeholder.container():
            with st.spinner(f"Fetching information about '{object_query}'..."):
                
                # AI response
                if user_question and gemini_model:
                    st.markdown("### 🤖 AI Assistant Response")
                    ai_response = chat_with_gemini(object_query, user_question, gemini_model)
                    st.success(ai_response)
    
    # Real-time detection
    if st.session_state.detection_active:
        status_placeholder.success("🟢 Detection Active")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            status_placeholder.error("❌ Failed to access webcam")
            st.session_state.detection_active = False
            return
        
        # Optimize camera settings for minimal lag
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Critical: reduce buffer lag
        
        frame_count = 0
        
        try:
            while st.session_state.detection_active:
                ret, frame = video_read(cap)
                if not ret:
                    # Clear buffer and continue
                    cap.grab()
                    continue
                
                frame_count += 1
                
                # Process only every N frames for performance
                if frame_count % process_every_n_frames == 0:
                    # Run YOLO detection with optimizations
                    results = model(
                        frame, 
                        conf=confidence, 
                        verbose=False, 
                        device='cpu',
                        half=False,  # Disable half precision for CPU
                        agnostic_nms=True  # Faster NMS
                    )
                    
                    detections = []
                    for result in results[0].boxes.data:
                        x1, y1, x2, y2, conf, cls = result
                        label = model.names[int(cls)]
                        detections.append((
                            [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                            float(conf),
                            label
                        ))
                    
                    # Update tracker (handles object movement)
                    tracks = tracker.update_tracks(detections, frame=frame)
                    
                    # Draw detections with tracking
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        
                        track_id = track.track_id
                        ltrb = track.to_ltrb()
                        x1, y1, x2, y2 = map(int, ltrb)
                        label = track.get_det_class()
                        
                        # Add to tracked objects
                        st.session_state.tracked_objects.add(label)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label with background
                        label_text = f"{label} #{track_id}"
                        (text_w, text_h), _ = cv2.getTextSize(
                            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                        )
                        cv2.rectangle(
                            frame, (x1, y1-text_h-10), 
                            (x1+text_w+10, y1), (0, 255, 0), -1
                        )
                        cv2.putText(
                            frame, label_text, (x1+5, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
                        )
                
                # Display frame immediately
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
        except Exception as e:
            status_placeholder.error(f"❌ Error during detection: {e}")
        finally:
            cap.release()
            st.session_state.detection_active = False
            status_placeholder.info("⚪ Detection Stopped")
    else:
        status_placeholder.info("⚪ Detection Inactive - Click 'Start Detection' to begin")
        video_placeholder.info("📷 Webcam feed will appear here when detection is started")

if __name__ == "__main__":
    main()