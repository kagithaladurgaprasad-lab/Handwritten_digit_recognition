import numpy as np
from tensorflow.keras.models import load_model


class DigitPredictor:

    def __init__(self, model_path="handwritten_digit_cnn.keras"):

        print("Loading CNN Model...")
        self.model = load_model(model_path)
        print("CNN Model Loaded Successfully!")

    def predict(self, image):
        """
        image shape expected: (1, 28, 28, 1)
        """
        if image is None:
            return "-", 0.0

        # Ensure datatype matches perfectly
        image = np.array(image, dtype=np.float32)

        prediction = self.model.predict(image, verbose=0)

        digit = int(np.argmax(prediction))

        confidence = float(np.max(prediction)) * 100

        return digit, confidence