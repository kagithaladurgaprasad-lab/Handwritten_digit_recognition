import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:

    def __init__(self):

        # Load MediaPipe Hand Landmarker Model
        base_options = python.BaseOptions(
            model_asset_path="hand_landmarker.task"
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        # Hand Connections
        self.connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17)
        ]

    # -------------------------
    # Detect Hand
    # -------------------------
    def detect(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(mp_image)

        return result

    # -------------------------
    # Draw Hand Landmarks
    # -------------------------
    def draw_landmarks(self, frame, result):

        if not result.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        for hand in result.hand_landmarks:

            # Draw Landmark Points
            for landmark in hand:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw Skeleton
            for start, end in self.connections:

                x1 = int(hand[start].x * w)
                y1 = int(hand[start].y * h)

                x2 = int(hand[end].x * w)
                y2 = int(hand[end].y * h)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

        return frame

    # -------------------------
    # Get Index Finger Tip
    # -------------------------
    def get_index_finger_tip(self, hand, frame):

        h, w, _ = frame.shape

        tip = hand[8]

        x = int(tip.x * w)
        y = int(tip.y * h)

        return x, y

    # -------------------------
    # Get All Landmark Coordinates
    # -------------------------
    def get_landmark_coordinates(self, hand, frame):

        h, w, _ = frame.shape

        landmarks = []

        for landmark in hand:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            landmarks.append((x, y))

        return landmarks