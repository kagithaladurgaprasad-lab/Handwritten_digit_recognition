import cv2
import numpy as np


class VirtualCanvas:

    def __init__(self, width=640, height=480):

        self.width = width
        self.height = height

        # Black Canvas
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Previous Point
        self.prev_x = None
        self.prev_y = None

        # Brush - Reduced from 12 to 8 to keep holes/loops clear for the AI
        self.brush_color = (255, 255, 255)
        self.brush_size = 8

        # Smoothing Factor
        # Change this line inside virtual_canvas.py
        self.smoothing = 0.1

        # Ignore very large jumps inside the canvas logic
        self.max_jump = 50

    # ----------------------------------
    # Smooth Drawing
    # ----------------------------------
    def draw(self, x, y):

        if self.prev_x is None:
            self.prev_x = x
            self.prev_y = y
            
            # Draw a small circle for the starting point
            cv2.circle(self.canvas, (x, y), self.brush_size // 2, self.brush_color, -1)
            return

        # Check for sudden massive tracking jumps
        distance = ((x - self.prev_x) ** 2 + (y - self.prev_y) ** 2) ** 0.5
        if distance > self.max_jump:
            self.prev_x = x
            self.prev_y = y
            return

        # Exponential smoothing
        smooth_x = int(self.prev_x * self.smoothing + x * (1 - self.smoothing))
        smooth_y = int(self.prev_y * self.smoothing + y * (1 - self.smoothing))

        # Connect previous point to current smoothed point
        cv2.line(
            self.canvas,
            (self.prev_x, self.prev_y),
            (smooth_x, smooth_y),
            self.brush_color,
            self.brush_size,
            cv2.LINE_AA
        )

        self.prev_x = smooth_x
        self.prev_y = smooth_y

    # ----------------------------------
    # Stop Drawing
    # ----------------------------------
    def stop_drawing(self):

        self.prev_x = None
        self.prev_y = None

    # ----------------------------------
    # Clear Canvas
    # ----------------------------------
    def clear(self):

        self.canvas[:] = 0

        self.prev_x = None
        self.prev_y = None

    # ----------------------------------
    # Get Canvas
    # ----------------------------------
    def get_canvas(self):

        return self.canvas

    # ----------------------------------
    # Overlay Canvas
    # ----------------------------------
    def overlay(self, frame, alpha=0.45):

        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)

        return cv2.add(frame_bg, canvas_fg)