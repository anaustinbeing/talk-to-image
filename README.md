# Ready to talk to image?

## Objective:

To interact with image using LangChain Agents; Ask questions about image like a human and get response about what the image is about.



## High-level flow:

![unnamed (1)](https://github.com/user-attachments/assets/b63e8ccb-55eb-4b9a-9126-d7ef7f60093b)

## Agents Architecture:

![unnamed (2)](https://github.com/user-attachments/assets/505abd03-f4cb-40d7-a0c0-e1b2d787d70f)

## Libraries and Frameworks used:

**1. Streamlit**

Role: Provides a simple, interactive interface for the user

Features:

Image uploader with supported formats (PNG, JPG)

Displays uploaded image for analysis

Text input and speech-to-text functionality for queries

Why Streamlit?: Easy integration with Python and quick UI development.

**2. Pillow (PIL)**

Role: Prepares the image for analysis

Features:

Opens and converts image to suitable formats for processing

Prepares the image for analysis by AI tools (captioning, description, and object detection)

Why Pillow?: It is lightweight and easy to integrate with Python-based projects.


**3. Langchain for Tool Integration**

Role: Provides a framework to integrate custom tools for processing the images

Custom Tools:

ImageCaptionTool: Generates short captions for images

ImageDescriptionTool: Produces detailed descriptions of images

ObjectDetectionTool: Detects objects present in the uploaded images

Why Langchain?: Simplifies the creation and orchestration of multi-modal AI tools.


## Tools

**1. ImageCaptionTool**

Model used: Salesforce/blip-image-captioning-large

Image Processing: Uses BlipProcessor to convert the image into model-readable input.

Model Loading: Loads the BlipForConditionalGeneration model for image captioning.

Caption Generation: The model generates a caption for the input image.

Output: Decodes and returns the caption in natural language.

**2. ImageDescriptionTool**

Model used: Salesforce/blip-image-captioning-large

Image Processing: BlipProcessor prepares the image as model input.

Model Loading: Utilizes BlipForConditionalGeneration for detailed image description.

Description Generation: The model produces an in-depth description of the image.

Output: Decodes and returns the description in natural language.

**3. ObjectDetectionTool**

Model used: facebook/detr-resnet-50

Image Processing: DetrImageProcessor converts the image for object detection.

Model Loading: Loads DetrForObjectDetection for identifying objects in the image.

Object Detection: Model processes the image and identifies objects with high confidence.

Output: Returns a list of detected objects with labels in natural language.

## How to start?

Clone the repo:

```
git clone https://github.com/anaustinbeing/talk-to-image.git
```

Change Directory into the talk-to-image:

```
cd talk-to-image
```

Create a virtual env in Python:

```
python -m venv venv
```

Activate virtual env (in Windows):

```
venv\Scripts\activate
```

Activate virtual env (in non-Windows):

```
source venv\bin\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Create `.streamlit` folder in the talk-to-image folder and add `secrets.toml` file to it with following content:

```
OPENAI_API_KEY='paste your openai api key here'
```

Start the application:

```
streamlit run app.py
```

Goto http://localhost:8501/ to access the application.

