class GestureController:

    def __init__(self):

        self.last_gesture = "NONE"
        self.counter = 0
        self.stable_frames = 5

    # -----------------------------------
    # Detect which fingers are up
    # -----------------------------------
    def fingers_up(self, hand):

        fingers = []

        # Thumb (Right Hand)
        fingers.append(1 if hand[4].x < hand[3].x else 0)

        # Index
        fingers.append(1 if hand[8].y < hand[6].y else 0)

        # Middle
        fingers.append(1 if hand[12].y < hand[10].y else 0)

        # Ring
        fingers.append(1 if hand[16].y < hand[14].y else 0)

        # Pinky
        fingers.append(1 if hand[20].y < hand[18].y else 0)

        return fingers

    # -----------------------------------
    # Raw Gesture Detection
    # -----------------------------------
    def detect_gesture(self, hand):

        fingers = self.fingers_up(hand)

        thumb, index, middle, ring, pinky = fingers

        # Debug (optional)
        print("Detected Fingers:", fingers)

        # ---------------- DRAW ----------------
        if index and not middle and not ring and not pinky:
            return "DRAW"

        # ---------------- STOP ----------------
        elif index and middle and not ring and not pinky:
            return "STOP"

        # ---------------- CLEAR ----------------
        # Ignore thumb because thumb detection is unstable
        elif (not index and
              not middle and
              not ring and
              not pinky):
            return "CLEAR"

        # ---------------- PREDICT ----------------
        elif index and middle and ring and pinky:
            return "PREDICT"

        return "NONE"

    # -----------------------------------
    # Stable Gesture Detection
    # -----------------------------------
    def get_gesture(self, hand):

        current = self.detect_gesture(hand)

        if current == self.last_gesture:
            self.counter += 1
        else:
            self.counter = 0
            self.last_gesture = current

        if self.counter >= self.stable_frames:
            return current

        return "NONE"