import torch

image_captioning_model = 'Salesforce/blip-image-captioning-large'
object_detecting_model = 'facebook/detr-resnet-50'
device = 'cuda' if torch.cuda.is_available() else 'cpu'