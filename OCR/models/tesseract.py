import pytesseract
from PIL import Image
from OCR.OCR import OCR


class Tesseract(OCR):
    def __init__(self):
        super().__init__("Tesseract")

    def predict(self, image: Image.Image) -> str:
        """
        Perform OCR on a PIL Image and return the recognized text.
        :param image: The PIL Image to perform OCR on.
        :return: The recognized text as a string.
        """
        # Perform OCR using Tesseract via pytesseract
        try:
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return ""
