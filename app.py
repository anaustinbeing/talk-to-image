import streamlit as st
from PIL import Image
from utils.image_processor import caption_image, describe_image
from utils.object_detector import detect_objects
from utils.agent_initializer import initialize_agent_with_tools
from streamlit_mic_recorder import speech_to_text

from langchain.tools import BaseTool

# Streamlit UI
st.title(':speaking_head_in_silhouette: :frame_with_picture: Talk To Image!')
uploaded_file = st.file_uploader(label='Upload images', type=['png', 'jpg'])

# Display uploaded image
if uploaded_file:
    if 'image' not in st.session_state or uploaded_file != st.session_state['image']:
        st.session_state['image'] = uploaded_file
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

# This is image caption tool - generates caption given an image.
class ImageCaptionTool(BaseTool):
    name: str = "Image captioner"
    description: str = (
        "Tool to generate a short caption of an image and read text from image."
    )

    def _run(self, query):
        image = Image.open(st.session_state['image']).convert('RGB')
        return caption_image(image)

    def _arun(self, query):
        raise NotImplementedError("This tool does not support async")
    
# This is image description tool - generates description of an image.
class ImageDescriptionTool(BaseTool):
    name: str = "Image descripter"
    description: str = (
        "Tool to generate a lengthy description of an image."
    )

    def _run(self, query):
        image = Image.open(st.session_state['image']).convert('RGB')
        return describe_image(image)

    def _arun(self, query):
        raise NotImplementedError("This tool does not support async")
    
# This is object detection tool - detects objects in the image.
class ObjectDetectionTool(BaseTool):
    name: str = "Object detector"
    description: str = "Tool to detect objects in image. This returns a list of all detected objects."

    def _run(self, img_path):
        image = Image.open(st.session_state['image']).convert('RGB')
        return detect_objects(image)

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

# Initialize agent
agent = initialize_agent_with_tools([ImageDescriptionTool(), ImageCaptionTool(), ObjectDetectionTool()], st.secrets['OPENAI_API_KEY'])

# Get user input through text or speech
user_query = st.text_input('Enter your query..')
user_speech_query = speech_to_text(
    language='en',
    start_prompt="Start recording",
    stop_prompt="Stop recording"
)
query = user_query if user_query else user_speech_query

# st.write('query: ', query)
# Process the image and query the agent
if uploaded_file and query:
    st.write(f'You asked: {query}')
    response = agent(f'{query}')
    st.write(response['output'])




