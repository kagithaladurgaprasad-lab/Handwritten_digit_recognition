import cv2
import numpy as np


class ImageProcessor:

    def __init__(self):
        self.size = 28

    def preprocess(self, canvas):

        # Convert to grayscale
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # Binary threshold
        _, thresh = cv2.threshold(
            gray,
            20,
            255,
            cv2.THRESH_BINARY
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        # Largest contour
        contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(contour)

        # Crop digit
        digit = thresh[y:y+h, x:x+w]

        # --- UPDATED PADDING STRATEGY ---
        # Add a 35% margin relative to the size to keep it perfectly centered 
        # away from the borders, matching training distribution.
        margin = int(max(w, h) * 0.35)
        size = max(w, h) + (margin * 2)

        square = np.zeros((size, size), dtype=np.uint8)

        x_offset = (size - w) // 2
        y_offset = (size - h) // 2

        square[
            y_offset:y_offset+h,
            x_offset:x_offset+w
        ] = digit

        # Resize to 28x28
        image = cv2.resize(
            square,
            (28, 28),
            interpolation=cv2.INTER_AREA
        )

        # Normalize 
        image = image.astype("float32") / 255.0

        # Form safe CNN shape explicitly
        image = np.expand_dims(image, axis=0)  
        image = np.expand_dims(image, axis=-1) 

        return image