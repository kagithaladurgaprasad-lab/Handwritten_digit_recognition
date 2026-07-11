import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np
import av
import time

# Import pipeline modules
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
        
        # Dimensions Setup
        self.target_width = 640
        self.target_height = 480
        self.last_processing_time = 0
        
        # Drawing layer matrix (White lines on black background)
        self.internal_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Vector points tracking
        self.last_x = None
        self.last_y = None
        
        # Internal processing state metrics
        self.prediction = "-"
        self.confidence = 0.0
        self.current_gesture = "NO HAND"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        if img.shape[1] != self.target_width or img.shape[0] != self.target_height:
            img = cv2.resize(img, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
        img = cv2.flip(img, 1)

        # Skip tracking overhead frames gracefully if cloud hits bandwidth ceilings
        current_time = time.time()
        if current_time - self.last_processing_time < 0.04:
            gray_canvas = cv2.cvtColor(self.internal_canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
            img[mask > 0] = [255, 255, 255]
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        self.last_processing_time = current_time

        # Run hand landmarks calculation
        result = self.tracker.detect(img)
        img = self.tracker.draw_landmarks(img, result)

        self.current_gesture = "NO HAND"

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            self.current_gesture = self.gesture_controller.get_gesture(hand)
            x, y = self.tracker.get_index_finger_tip(hand, img)

            # Draw location confirmation dot
            cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

            if self.current_gesture == "DRAW":
                if self.last_x is not None:
                    cv2.line(self.internal_canvas, (self.last_x, self.last_y), (x, y), (255, 255, 255), 10)
                else:
                    cv2.circle(self.internal_canvas, (x, y), 5, (255, 255, 255), -1)
                self.last_x = x
                self.last_y = y

            elif self.current_gesture == "PREDICT":
                self.last_x = None
                self.last_y = None
                processed_img = self.processor.preprocess(self.internal_canvas)
                if processed_img is not None:
                    pred, conf = self.predictor.predict(processed_img)
                    self.prediction = str(pred)
                    self.confidence = float(conf)

            elif self.current_gesture == "CLEAR":
                self.internal_canvas.fill(0)
                self.prediction = "-"
                self.confidence = 0.0
                self.last_x = None
                self.last_y = None
            else:
                self.last_x = None
                self.last_y = None
        else:
            self.last_x = None
            self.last_y = None

        # Render white vectors onto live monitor feed image array
        gray_canvas = cv2.cvtColor(self.internal_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
        img[mask > 0] = [255, 255, 255]

        # Draw HUD overlays on frame matrix to eliminate text layout shifts
        cv2.putText(img, f"GESTURE: {self.current_gesture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, f"PREDICTION: {self.prediction} ({self.confidence:.1f}%)", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------
# Unified Video Stream Component
# -----------------------------
st.write("Position your hand inside the frame to draw. The state engine output is burned directly into the monitor video.")

ctx = webrtc_streamer(
    key="air-drawing-v13-stable",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {"width": 640, "height": 480, "frameRate": 30},
        "audio": False
    },
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_html_attrs={
        "style": {"width": "100%", "max-width": "720px", "border": "2px solid #333", "border-radius": "8px", "margin": "0 auto"},
        "controls": False,
        "autoPlay": True,
        "playsInline": True,
    }
)

st.write("---")
st.markdown("""
### System Gestures:
* 🖐️ **DRAW**: Move index finger to paint white lines on screen.
* ✊ **CLEAR**: Make a closed fist to wipe the canvas instantly.
* ✌️ **PREDICT**: Hold up a gesture to evaluate drawings against the AI model.
""")

# Sidebar auxiliary reset button
st.sidebar.title("Controls")
if st.sidebar.button("🧼 Emergency Reset Canvas", use_container_width=True):
    if ctx.video_processor:
        ctx.video_processor.internal_canvas.fill(0)
        ctx.video_processor.prediction = "-"
        ctx.video_processor.confidence = 0.0
        ctx.video_processor.last_x = None
        ctx.video_processor.last_y = None
