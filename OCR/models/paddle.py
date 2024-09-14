from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from OCR.OCR import OCR


class PaddleOCREngine(OCR):
    def __init__(self):
        super().__init__("PaddleOCR")
        # Initialize PaddleOCR with default configuration (supports English and Chinese by default)
        self.ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

    def predict(self, image: Image.Image) -> str:
        """
        Perform OCR on a PIL Image using PaddleOCR and return the recognized text.
        :param image: The PIL Image to perform OCR on.
        :return: The recognized text as a string.
        """
        # Convert the PIL image to a NumPy array (format expected by PaddleOCR)
        image_np = np.array(image)

        try:
            # Perform OCR using PaddleOCR
            result = self.ocr_model.ocr(image_np, cls=True)

            # Extract the recognized text from the result
            text = [line[1][0] for line in result[0]]  # Extract the text part from the OCR result
            return " ".join(text)  # Join all recognized text lines into a single string
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return ""
