import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from utils.config import image_captioning_model, device

# Initialize model and processor
processor = BlipProcessor.from_pretrained(image_captioning_model)
model = BlipForConditionalGeneration.from_pretrained(image_captioning_model).to(device)

def caption_image(image: Image):
    """Processes an image and generates a caption."""
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def describe_image(image: Image):
    """Processes an image and describes the image."""
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=500)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description



