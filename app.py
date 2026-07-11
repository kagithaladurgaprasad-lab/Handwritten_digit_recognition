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

# =========================================================
# Page config + light theming
# =========================================================
st.set_page_config(page_title="Air Digit Recognition", page_icon="🖐️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .status-card {
        background: #111827;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: white;
        margin-bottom: 0.6rem;
        border: 1px solid #2d3748;
    }
    .status-label {font-size: 0.8rem; opacity: 0.7; letter-spacing: 0.05em; text-transform: uppercase;}
    .status-value {font-size: 1.6rem; font-weight: 700; margin-top: 0.15rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🖐️ Air Digit Recognition Studio")
st.caption("Draw a digit in the air with your index finger — no mouse, no touch.")

# =========================================================
# Sidebar controls
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")

    quality_preset = st.select_slider(
        "Performance preset",
        options=["Ultra low power", "Battery saver", "Balanced", "High quality"],
        value="Battery saver",
        help="Lower presets trade video sharpness for a smoother, less laggy stream — "
             "recommended for cloud deployments. Start with 'Ultra low power' if things "
             "are still laggy.",
    )
    PRESETS = {
        "Ultra low power": dict(width=320, height=240, fps=10, detect_scale=0.4, process_every=3),
        "Battery saver":   dict(width=480, height=360, fps=15, detect_scale=0.5, process_every=2),
        "Balanced":        dict(width=640, height=480, fps=20, detect_scale=0.5, process_every=1),
        "High quality":    dict(width=640, height=480, fps=30, detect_scale=0.75, process_every=1),
    }
    cfg = PRESETS[quality_preset]

    show_skeleton = st.checkbox(
        "Show hand skeleton overlay", value=(quality_preset in ("Balanced", "High quality")),
        help="Drawing the landmark skeleton every frame costs extra time — turning it off "
             "helps on slow connections.",
    )
    line_thickness = st.slider("Line thickness", 4, 20, 10)
    use_turn = st.checkbox(
        "Use TURN relay (recommended on cloud)",
        value=True,
        help="STUN alone frequently fails on Streamlit Community Cloud's network, causing "
             "the exact lag/freeze you're seeing. TURN relays the video and fixes it.",
    )
    adaptive_skip = st.checkbox(
        "Adaptive frame skipping", value=True,
        help="Automatically skips more frames if the server is running behind, so the "
             "video doesn't build up a growing backlog of stale frames.",
    )

    st.divider()
    st.subheader("Live status")
    status_placeholder = st.empty()

    st.divider()
    reset_clicked = st.button("🧼 Emergency Reset Canvas", use_container_width=True)


# =========================================================
# ICE / RTC configuration
# =========================================================
def get_rtc_configuration(enable_turn: bool):
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    if enable_turn:
        # Free public TURN relay (Open Relay Project / metered.ca). Fine for demos & light
        # traffic. For production, swap in your own TURN credentials (Twilio, Cloudflare,
        # metered.ca paid tier, etc.) — public relays can be rate-limited.
        ice_servers.append(
            {
                "urls": [
                    "turn:global.relay.metered.ca:80",
                    "turn:global.relay.metered.ca:443",
                    "turn:global.relay.metered.ca:443?transport=tcp",
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            }
        )
    return {"iceServers": ice_servers}


# =========================================================
# Video Processing Worker
# =========================================================
class VideoProcessor:
    def __init__(self):
        self.tracker = HandTracker()
        self.gesture_controller = GestureController()
        self.processor = ImageProcessor()
        self.predictor = DigitPredictor()

        self.target_width = cfg["width"]
        self.target_height = cfg["height"]
        self.detect_scale = cfg["detect_scale"]      # run the hand detector on a smaller frame
        self.process_every = cfg["process_every"]     # baseline: only run detection every Nth frame
        self.adaptive_skip = adaptive_skip
        self.show_skeleton = show_skeleton
        self._frame_count = 0
        self._proc_time_ema = 1.0 / cfg["fps"]        # running estimate of detection cost (seconds)

        self.internal_canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

        # Cached binary mask of the canvas — only rebuilt when the canvas actually changes,
        # instead of on every single frame.
        self._canvas_dirty = True
        self._cached_mask = None

        self.last_x = None
        self.last_y = None

        self.prediction = "-"
        self.confidence = 0.0
        self.current_gesture = "NO HAND"
        self.fps = 0.0
        self._last_tick = time.time()

    # -- helpers -------------------------------------------------
    def _get_canvas_mask(self):
        if self._canvas_dirty or self._cached_mask is None:
            gray = cv2.cvtColor(self.internal_canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            self._cached_mask = mask > 0
            self._canvas_dirty = False
        return self._cached_mask

    def _draw_hud(self, img):
        cv2.putText(img, f"GESTURE: {self.current_gesture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, f"PREDICTION: {self.prediction} ({self.confidence:.1f}%)", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img

    def reset_canvas(self):
        self.internal_canvas.fill(0)
        self.prediction = "-"
        self.confidence = 0.0
        self.last_x = None
        self.last_y = None
        self._canvas_dirty = True

    # -- main callback --------------------------------------------
    # NOTE: recv_queued (instead of recv) is what stops lag from *accumulating*. With a plain
    # recv(), if the detector takes longer than the frame interval, streamlit-webrtc queues
    # incoming frames and processes them one by one — the delay keeps growing and the video
    # falls further and further behind. recv_queued gives us the whole backlog at once; we
    # process only the newest frame and drop the rest, so the stream always shows "now",
    # never a growing backlog of "a few seconds ago".
    async def recv_queued(self, frames):
        frame = frames[-1]  # always take the most recent frame, drop any older queued ones
        img = frame.to_ndarray(format="bgr24")

        if img.shape[1] != self.target_width or img.shape[0] != self.target_height:
            img = cv2.resize(img, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
        img = cv2.flip(img, 1)

        self._frame_count += 1

        # Adaptive skip: if detection has been taking longer than the frame budget, skip
        # detection on proportionally more frames instead of falling behind.
        effective_skip = self.process_every
        if self.adaptive_skip:
            frame_budget = 1.0 / max(self.fps, 1.0) if self.fps > 0 else (1.0 / 15)
            if self._proc_time_ema > frame_budget:
                effective_skip = max(self.process_every, int(self._proc_time_ema / frame_budget) + 1)

        run_detection = (self._frame_count % effective_skip == 0)

        if run_detection:
            t0 = time.time()
            # Detect on a downscaled copy for speed. HandTracker/get_index_finger_tip are
            # expected to return normalized (0-1) landmark coordinates, so we still draw and
            # read positions using the full-resolution `img` — only the (expensive) detector
            # call itself runs on the smaller frame.
            if self.detect_scale < 1.0:
                small = cv2.resize(img, None, fx=self.detect_scale, fy=self.detect_scale,
                                    interpolation=cv2.INTER_LINEAR)
                result = self.tracker.detect(small)
            else:
                result = self.tracker.detect(img)

            if self.show_skeleton:
                img = self.tracker.draw_landmarks(img, result)

            # track how expensive detection actually was, to drive adaptive skipping
            proc_dt = time.time() - t0
            self._proc_time_ema = 0.8 * self._proc_time_ema + 0.2 * proc_dt

            self.current_gesture = "NO HAND"

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                self.current_gesture = self.gesture_controller.get_gesture(hand)
                x, y = self.tracker.get_index_finger_tip(hand, img)
                cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

                if self.current_gesture == "DRAW":
                    if self.last_x is not None:
                        cv2.line(self.internal_canvas, (self.last_x, self.last_y), (x, y),
                                  (255, 255, 255), line_thickness)
                    else:
                        cv2.circle(self.internal_canvas, (x, y), line_thickness // 2, (255, 255, 255), -1)
                    self.last_x, self.last_y = x, y
                    self._canvas_dirty = True

                elif self.current_gesture == "PREDICT":
                    self.last_x = None
                    self.last_y = None
                    processed_img = self.processor.preprocess(self.internal_canvas)
                    if processed_img is not None:
                        pred, conf = self.predictor.predict(processed_img)
                        self.prediction = str(pred)
                        self.confidence = float(conf)

                elif self.current_gesture == "CLEAR":
                    self.reset_canvas()
                else:
                    self.last_x = None
                    self.last_y = None
            else:
                self.last_x = None
                self.last_y = None

        # Composite the (cached) canvas mask onto the live frame every frame, cheaply.
        mask = self._get_canvas_mask()
        img[mask] = [255, 255, 255]
        img = self._draw_hud(img)

        # rough FPS tracking for the status panel
        now = time.time()
        dt = now - self._last_tick
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self._last_tick = now

        return [av.VideoFrame.from_ndarray(img, format="bgr24")]


# =========================================================
# Layout: video + instructions
# =========================================================
video_col, info_col = st.columns([2, 1])

with video_col:
    ctx = webrtc_streamer(
        key="air-drawing-v15-optimized",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": cfg["width"]},
                "height": {"ideal": cfg["height"]},
                "frameRate": {"ideal": cfg["fps"], "max": cfg["fps"]},
            },
            "audio": False,
        },
        async_processing=True,
        rtc_configuration=get_rtc_configuration(use_turn),
        video_html_attrs={
            "style": {
                "width": "100%",
                "max-width": "720px",
                "border": "2px solid #333",
                "border-radius": "12px",
                "margin": "0 auto",
                "display": "block",
            },
            "controls": False,
            "autoPlay": True,
            "playsInline": True,
        },
    )

with info_col:
    st.subheader("System Gestures")
    st.markdown(
        """
- 🖐️ **DRAW** — move index finger to paint
- ✊ **CLEAR** — closed fist wipes the canvas
- ✌️ **PREDICT** — hold gesture to run the model
"""
    )
    st.info(
        "If the video still feels slow after switching to **Battery saver** + **TURN relay**, "
        "the bottleneck is likely CPU on Streamlit Cloud's shared instance rather than the network — "
        "consider a smaller/quantized model in `digit_predictor.py`."
    )

# Handle sidebar reset
if reset_clicked and ctx.video_processor:
    ctx.video_processor.reset_canvas()

# =========================================================
# Live status panel (polls the processor thread while streaming)
# =========================================================
if ctx.state.playing:
    while ctx.state.playing:
        vp = ctx.video_processor
        if vp is not None:
            status_placeholder.markdown(
                f"""
<div class="status-card">
  <div class="status-label">Gesture</div>
  <div class="status-value">{vp.current_gesture}</div>
</div>
<div class="status-card">
  <div class="status-label">Prediction</div>
  <div class="status-value">{vp.prediction} <span style="font-size:0.9rem; opacity:0.7;">({vp.confidence:.1f}%)</span></div>
</div>
<div class="status-card">
  <div class="status-label">Processing FPS</div>
  <div class="status-value">{vp.fps:.1f}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        time.sleep(0.3)
else:
    status_placeholder.info("Start the camera to see live status.")
