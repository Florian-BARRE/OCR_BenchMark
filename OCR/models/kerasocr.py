import keras_ocr
from PIL import Image
import numpy as np
from OCR.OCR import OCR


class KerasOCR(OCR):
    def __init__(self):
        super().__init__("KerasOCR")
        # Initialize the Keras-OCR pipeline (which includes both text detection and recognition)
        try:
            self.pipeline = keras_ocr.pipeline.Pipeline()
        except Exception as e:
            print(f"Error initializing Keras-OCR pipeline: {e}")
            self.pipeline = None  # Set to None if initialization fails

    def predict(self, image: Image.Image) -> str:
        """
        Perform OCR on a PIL Image using Keras-OCR and return the recognized text.
        :param image: The PIL Image to perform OCR on.
        :return: The recognized text as a string.
        """
        # Ensure the pipeline is initialized
        if self.pipeline is None:
            print("Keras-OCR pipeline is not initialized.")
            return ""

        # Convert PIL Image to numpy array (since Keras-OCR works with numpy arrays)
        image_np = np.array(image)

        # Perform OCR on the image
        try:
            prediction_groups = self.pipeline.recognize([image_np])

            # Extract text from the predictions
            recognized_text = ' '.join([text for _, text in prediction_groups[0]])
            return recognized_text
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return ""
