import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from PIL import Image
from OCR.OCR import OCR


class PhiOCR(OCR):
    def __init__(self):
        super().__init__("PhiOCR")

        # Load model and processor from Hugging Face Hub
        model_id = "yifeihu/TB-OCR-preview-0.1"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model with memory-efficient options
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True)  # Optional: Load model in 4-bit to save memory
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=16)

    def predict(self, image: Image.Image) -> str:
        """
        Perform OCR on a PIL Image using Phi-3.5-vision-instruct (Hugging Face model) and return the recognized text.
        :param image: The PIL Image to perform OCR on.
        :return: The recognized text as a string.
        """
        question = "Convert the text to markdown format."  # Required prompt
        prompt_message = [{
            'role': 'user',
            'content': f'<|image_1|>\n{question}',
        }]

        # Apply processor and send inputs to device
        prompt = self.processor.tokenizer.apply_chat_template(prompt_message, tokenize=False,
                                                              add_generation_prompt=True)
        inputs = self.processor(prompt, [image], return_tensors="pt").to(self.device)

        # Generation configuration
        generation_args = {
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "do_sample": False
        }

        # Generate text based on the image input
        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id,
                                           **generation_args)
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

        # Decode and clean up the response
        response = \
        self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = response.split("<image_end>")[0]  # Remove any unwanted tokens

        return response
