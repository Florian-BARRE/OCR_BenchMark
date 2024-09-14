import easyocr
from PIL import Image
from OCR.OCR import OCR
import numpy as np


class EasyOCR(OCR):
    def __init__(self):
        super().__init__("EasyOCR")
        self.reader = easyocr.Reader(['en'])  # Initialize EasyOCR with the language(s) you need

    def predict(self, image: Image.Image) -> str:
        """
        Perform OCR on a PIL Image using EasyOCR and return the recognized text.
        :param image: The PIL Image to perform OCR on.
        :return: The recognized text as a string.
        """
        # Convert the PIL image to OpenCV format (numpy array)
        image_np = self._pil_to_cv2(image)

        try:
            # Perform OCR using EasyOCR
            result = self.reader.readtext(image_np, detail=0)
            text = ' '.join(result)  # Join all the text snippets into a single string
            return text
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return ""

    def _pil_to_cv2(self, image: Image.Image):
        """
        Convert a PIL Image to an OpenCV (numpy array) format, required by EasyOCR.
        :param image: The PIL Image to convert.
        :return: The converted OpenCV image (numpy array).
        """
        return np.array(image)
