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
        
        # Consistent Canvas Size Setup
        self.target_width = 640
        self.target_height = 480
        self.last_processing_time = 0
        
        # Drawing canvas layer (White lines on black background)
        self.internal_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Coordinates tracking to prevent broken lines
        self.last_x = None
        self.last_y = None
        
        # Thread communication variables
        self.prediction = "-"
        self.confidence = 0.0
        self.current_gesture = "NO HAND"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Standardize dimension resolution
        if img.shape[1] != self.target_width or img.shape[0] != self.target_height:
            img = cv2.resize(img, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
        img = cv2.flip(img, 1)

        # Optimization: Limit hand-tracking calculations to 25 FPS to maximize cloud performance
        current_time = time.time()
        if current_time - self.last_processing_time < 0.04:
            # Still display the persistent drawing lines even if we drop tracking on this frame
            gray_canvas = cv2.cvtColor(self.internal_canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
            img[mask > 0] = [255, 255, 255]
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        self.last_processing_time = current_time

        # Run MediaPipe hand-tracker
        result = self.tracker.detect(img)
        img = self.tracker.draw_landmarks(img, result)

        self.current_gesture = "NO HAND"

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            self.current_gesture = self.gesture_controller.get_gesture(hand)
            x, y = self.tracker.get_index_finger_tip(hand, img)

            # Draw visual tracking pointer dot
            cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

            if self.current_gesture == "DRAW":
                if self.last_x is not None:
                    # Draw solid connected vectors
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

        # Mask operations to render drawings cleanly into live view
        gray_canvas = cv2.cvtColor(self.internal_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
        img[mask > 0] = [255, 255, 255]

        # Draw status text directly onto the video array matrix frame
        cv2.putText(img, f"Gesture: {self.current_gesture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------
# Render UI Layout Grid
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed Window")
    ctx = webrtc_streamer(
        key="air-drawing-v11-perfect",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {"width": 640, "height": 480, "frameRate": 30},
            "audio": False
        },
        async_processing=True,
        video_html_attrs={
            "style": {"width": "100%", "border": "2px solid #333", "border-radius": "8px"},
            "controls": False,
            "autoPlay": True,
            "playsInline": True,
        }
    )

with col2:
    st.subheader("AI Prediction Analysis")
    
    # FIX BLINKING: Isolated component fragment container for text layout rendering
    @st.fragment(run_every=0.1)
    def render_metrics_dashboard():
        pred_val = "-"
        conf_val = 0.0
        current_g = "NO HAND"
        
        if ctx.video_processor:
            pred_val = ctx.video_processor.prediction
            conf_val = ctx.video_processor.confidence
            current_g = ctx.video_processor.current_gesture

        st.metric(label="Predicted Digit Label", value=pred_val)
        st.metric(label="Model Confidence Match", value=f"{conf_val:.2f}%")
        st.info(f"Current Gesture Tracking: **{current_g}**")

    # Initialize layout fragment block
    render_metrics_dashboard()
    
    st.write("---")
    st.markdown("""
    ### System Gestures:
    * 🖐️ **DRAW**: Move index finger to paint white lines on screen.
    * ✊ **CLEAR**: Make a closed fist to wipe the canvas instantly.
    * ✌️ **PREDICT**: Hold up a gesture to evaluate drawings against the AI model.
    """)

# Sidebar manual clear canvas controller button
st.sidebar.title("Controls & Status")
if st.sidebar.button("🧼 Clear Canvas", use_container_width=True, key="canvas_clear_btn_v11"):
    if ctx.video_processor:
        ctx.video_processor.internal_canvas.fill(0)
        ctx.video_processor.prediction = "-"
        ctx.video_processor.confidence = 0.0
        ctx.video_processor.last_x = None
        ctx.video_processor.last_y = None
