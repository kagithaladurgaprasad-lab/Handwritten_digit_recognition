import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np
import av
import time

# Import your pipeline modules
from hand_tracker import HandTracker
from gesture_controller import GestureController
from virtual_canvas import VirtualCanvas
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
        self.canvas = VirtualCanvas()
        self.processor = ImageProcessor()
        self.predictor = DigitPredictor()
        
        # Performance Tuning: Lower resolution logic to reduce internet/cloud lag
        self.target_width = 640
        self.target_height = 480
        
        # State trackers
        self.last_x = None
        self.last_y = None
        self.max_jump = 50  
        self.last_processing_time = 0
        
        # Smooth line retention frame buffers
        self.frames_since_last_hand = 0
        self.max_missing_frames_buffer = 4 
        
        # Thread communication commands
        self.force_clear_canvas = False
        
        # Thread metrics tracking
        self.prediction = "-"
        self.confidence = 0.0
        self.current_gesture = "NO HAND"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Optimization 1: Downscale incoming video to 640x480 instantly to save cloud CPU bandwidth
        if img.shape[1] > self.target_width or img.shape[0] > self.target_height:
            img = cv2.resize(img, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
            
        img = cv2.flip(img, 1)

        # Process forced clear requests immediately
        if self.force_clear_canvas:
            self.canvas.clear()
            self.prediction = "-"
            self.confidence = 0.0
            self.last_x = None
            self.last_y = None
            self.force_clear_canvas = False 

        # Optimization 2: Intelligent Frame Dropping 
        # Limits processing strictly to ~22 FPS to prevent remote background container thread locking
        current_time = time.time()
        if current_time - self.last_processing_time < 0.045:
            output_frame = self.canvas.overlay(img)
            return av.VideoFrame.from_ndarray(output_frame, format="bgr24")
        
        self.last_processing_time = current_time

        # Execute tracking pipeline modules
        result = self.tracker.detect(img)
        img = self.tracker.draw_landmarks(img, result)

        if result.hand_landmarks:
            self.frames_since_last_hand = 0 
            hand = result.hand_landmarks[0]
            self.current_gesture = self.gesture_controller.get_gesture(hand)
            x, y = self.tracker.get_index_finger_tip(hand, img)

            cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

            if self.current_gesture == "DRAW":
                if self.last_x is not None:
                    distance = ((x - self.last_x) ** 2 + (y - self.last_y) ** 2) ** 0.5
                    if distance < self.max_jump:
                        self.canvas.draw(x, y)
                    else:
                        self.canvas.draw(x, y)
                else:
                    self.canvas.draw(x, y)
                
                self.last_x = x
                self.last_y = y

            elif self.current_gesture == "PREDICT":
                processed_img = self.processor.preprocess(self.canvas.get_canvas())
                if processed_img is not None:
                    pred, conf = self.predictor.predict(processed_img)
                    self.prediction = str(pred)
                    self.confidence = float(conf)
                
                self.canvas.stop_drawing()
                self.last_x = None
                self.last_y = None

            elif self.current_gesture == "CLEAR":
                self.canvas.clear()
                self.prediction = "-"
                self.confidence = 0.0
                self.canvas.stop_drawing()
                self.last_x = None
                self.last_y = None
            
            else:
                self.canvas.stop_drawing()
                self.last_x = None
                self.last_y = None
        else:
            self.frames_since_last_hand += 1
            self.current_gesture = "NO HAND"
            
            if self.frames_since_last_hand >= self.max_missing_frames_buffer:
                self.canvas.stop_drawing()
                self.last_x = None
                self.last_y = None

        # Build composite final output video frame
        output_frame = self.canvas.overlay(img)
        cv2.putText(output_frame, f"Gesture: {self.current_gesture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(output_frame, format="bgr24")

# -----------------------------
# Render UI & Stream Control Blocks
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed Window")
    
    # Optimization 3: Constraint passing to tell your browser to output lower byte streams directly
    ctx = webrtc_streamer(
        key="air-drawing-v8-cloud",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 24}
            },
            "audio": False
        },
        async_processing=True,
    )

with col2:
    st.subheader("AI Prediction Analysis")
    
    metric_slot_1 = st.empty()
    metric_slot_2 = st.empty()
    
    st.write("---")
    st.markdown("""
    ### How to use:
    1. Click **START** to establish a secure browser connection.
    2. Raise your hand within frame boundaries.
    3. Transition into **DRAW** gesture parameters to write.
    4. Transition into **PREDICT** parameters to run classification matrices.
    """)

    metric_slot_1.metric(label="Predicted Digit Label", value="-")
    metric_slot_2.metric(label="Model Confidence Match", value="0.00%")

    st.sidebar.title("Controls & Status")
    clear_clicked = st.sidebar.button("🧼 Clear Canvas", use_container_width=True, key="canvas_clear_btn_v8")

    # Lightweight metric refresh loop
    while ctx.video_processor:
        if clear_clicked:
            ctx.video_processor.force_clear_canvas = True
            clear_clicked = False 

        pred_val = ctx.video_processor.prediction
        conf_val = ctx.video_processor.confidence
        
        metric_slot_1.metric(label="Predicted Digit Label", value=pred_val)
        metric_slot_2.metric(label="Model Confidence Match", value=f"{conf_val:.2f}%")
            
        time.sleep(0.08)  # Optimization 4: Polling dropped to 12Hz to prevent main thread blocking
