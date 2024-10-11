from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
from utils.config import object_detecting_model, device

processor = DetrImageProcessor.from_pretrained(object_detecting_model)
model = DetrForObjectDetection.from_pretrained(object_detecting_model)

def detect_objects(image: Image):
    """Processes an image and detects objects in the image."""
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    detections = ', '.join(model.config.id2label[int(label)] for label in results['labels'])
    return detections