from abc import ABC, abstractmethod
from PIL import Image


class OCR(ABC):
    def __init__(self, model_name) -> None:
        self.model_name = model_name

    @abstractmethod
    def predict(self, image: Image.Image) -> str:
        raise NotImplementedError("Predict method not implemented !")
