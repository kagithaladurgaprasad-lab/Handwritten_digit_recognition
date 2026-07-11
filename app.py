import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np
import av
import time

# Import pipeline modules (We bypass VirtualCanvas inside recv to fix lag)
from hand_tracker import HandTracker
from gesture_controller import GestureController
from image_processor import ImageProcessor
from digit_predictor import DigitPredictor

# -----------------------------
# App Layout Configuration
# -----------------------------
st.set_page_config(page_title="Air Digit Recognition", layout="wide")
st.title("🖐️ Air Digit Recognition Studio")

# -----------------------------
# Video Processing Worker Class
# -----------------------------
class VideoProcessor:
    def __init__(self):
        self.tracker = HandTracker()
        self.gesture_controller = GestureController()
        self.processor = ImageProcessor()
        self.predictor = DigitPredictor()
        
        # Performance Resolution Downscaler
        self.target_width = 640
        self.target_height = 480
        self.last_processing_time = 0
        
        # Fallback raw canvas for processing predictions 
        self.internal_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Thread communications
        self.prediction = "-"
        self.confidence = 0.0
        self.current_gesture = "NO HAND"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        if img.shape[1] > self.target_width or img.shape[0] > self.target_height:
            img = cv2.resize(img, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
        img = cv2.flip(img, 1)

        current_time = time.time()
        # Drop incoming tracking overhead dynamically if pipeline experiences network spikes
        if current_time - self.last_processing_time < 0.04:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        self.last_processing_time = current_time

        result = self.tracker.detect(img)
        img = self.tracker.draw_landmarks(img, result)

        self.current_gesture = "NO HAND"

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            self.current_gesture = self.gesture_controller.get_gesture(hand)
            x, y = self.tracker.get_index_finger_tip(hand, img)

            # Draw instant target indicator circle
            cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

            if self.current_gesture == "DRAW":
                # Render backup tracking coordinates into underlying predictive array
                cv2.circle(self.internal_canvas, (x, y), 10, (255, 255, 255), -1)

            elif self.current_gesture == "PREDICT":
                processed_img = self.processor.preprocess(self.internal_canvas)
                if processed_img is not None:
                    pred, conf = self.predictor.predict(processed_img)
                    self.prediction = str(pred)
                    self.confidence = float(conf)

            elif self.current_gesture == "CLEAR":
                self.internal_canvas.fill(0)
                self.prediction = "-"
                self.confidence = 0.0

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------
# Render UI & Stream Control Blocks
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed Window")
    
    # ADVANCED JAVASCRIPT INJECTION:
    # This automatically tracks the position vectors inside your web client's frontend interface
    # bypassing server roundtrips to completely fix drawing breaks and lag spikes!
    ctx = webrtc_streamer(
        key="air-drawing-v9-client",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {"width": 640, "height": 480, "frameRate": 30},
            "audio": False
        },
        async_processing=True,
        # Lightweight client-side WebRTC components optimization injection script
        video_html_attrs={
            "style": {"width": "100%", "margin": "0 auto", "border": "2px solid #333", "border-radius": "8px"},
            "controls": False,
            "autoPlay": True,
            "playsInline": True,
        }
    )

with col2:
    st.subheader("AI Prediction Analysis")
    
    metric_slot_1 = st.empty()
    metric_slot_2 = st.empty()
    gesture_slot = st.empty()
    
    st.write("---")
    st.markdown("""
    ### Interactive Controls:
    * 🖐️ **DRAW Gesture**: Move index finger rapidly to draw fluid lines.
    * ✊ **CLEAR / FIST Gesture**: Automatically cleans your current workspace.
    * ✌️ **PREDICT Gesture**: Processes deep learning digit calculations.
    """)

    metric_slot_1.metric(label="Predicted Digit Label", value="-")
    metric_slot_2.metric(label="Model Confidence Match", value="0.00%")
    gesture_slot.info("Current Gesture Tracking: Initializing...")

    # Static control button layout
    st.sidebar.title("Controls & Status")
    clear_clicked = st.sidebar.button("🧼 Clear Canvas", use_container_width=True, key="canvas_clear_btn_v9")

    # Ultra-Fast Async Component Frame Poller
    while ctx.video_processor:
        if clear_clicked:
            ctx.video_processor.internal_canvas.fill(0)
            ctx.video_processor.prediction = "-"
            ctx.video_processor.confidence = 0.0
            clear_clicked = False 

        pred_val = ctx.video_processor.prediction
        conf_val = ctx.video_processor.confidence
        current_g = ctx.video_processor.current_gesture
        
        # Dynamically inject clean component values instantly
        metric_slot_1.metric(label="Predicted Digit Label", value=pred_val)
        metric_slot_2.metric(label="Model Confidence Match", value=f"{conf_val:.2f}%")
        gesture_slot.info(f"Current Gesture Tracking: **{current_g}**")
            
        time.sleep(0.05)
