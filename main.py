from dataset_manager import DatasetManager
from OCR_validator import OCRValidator
from OCR import Tesseract, EasyOCR, PaddleOCREngine, PhiOCR, KerasOCR

dataset_manager = DatasetManager("Salesforce/blip3-ocr-200m", num_rows=10)
validator = OCRValidator()

if __name__ == "__main__":
    OCRs = [EasyOCR(), PaddleOCREngine(), KerasOCR()]

    for image, captions, progress in dataset_manager.get_test():

        for ocr in OCRs:
            prediction = ocr.predict(image)
            validator.evaluate(ocr.model_name, prediction, captions)

        print(f"Progress: {progress*100:.2f}%")

    validator.display_scores()
