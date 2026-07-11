import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode


class HandTracker:

    def __init__(self):
        # Load MediaPipe Hand Landmarker Model Optimized for Video Tracking Streams
        base_options = python.BaseOptions(
            model_asset_path="hand_landmarker.task"
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,  # CRITICAL: Enables rapid temporal cross-frame tracking
            num_hands=1,                     # Locks calculation resources to a single target
            min_hand_detection_confidence=0.4, # Dropped slightly for much faster initial detection handshakes
            min_tracking_confidence=0.5
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        # Hand Connections (Static Definition)
        self.connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17)
        ]

    # -------------------------
    # Detect Hand (Optimized for Video Frames)
    # -------------------------
    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Generate a unique timestamp in milliseconds for running video mode tracking
        timestamp_ms = int(time.time() * 1000)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        return result

    # -------------------------
    # Draw Hand Landmarks (Optimized: No Redundant Math)
    # -------------------------
    def draw_landmarks(self, frame, result):
        if not result.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        for hand in result.hand_landmarks:
            # Step 1: Pre-compute pixels ONCE instead of recalculating for dots and lines
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            # Step 2: Draw Skeleton Connections fast using pre-computed index mapping
            for start, end in self.connections:
                cv2.line(
                    frame,
                    coords[start],
                    coords[end],
                    (255, 0, 0),
                    2
                )

            # Step 3: Draw Landmark Points
            for pt in coords:
                cv2.circle(
                    frame,
                    pt,
                    4,
                    (0, 255, 0),
                    -1
                )

        return frame

    # -------------------------
    # Get Index Finger Tip
    # -------------------------
    def get_index_finger_tip(self, hand, frame):
        h, w, _ = frame.shape
        tip = hand[8]
        return int(tip.x * w), int(tip.y * h)

    # -------------------------
    # Get All Landmark Coordinates
    # -------------------------
    def get_landmark_coordinates(self, hand, frame):
        h, w, _ = frame.shape
        return [(int(lm.x * w), int(lm.y * h)) for lm in hand]
