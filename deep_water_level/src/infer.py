# This script defines a Gradio interface to run the deep water level model on an input image.
# It takes an image as input and outputs the predicted water level.
#
# In order to run it:
# - Run training or get a model, save it as model.pth at the root of the project
# - Run this script, then open the Gradio app at http://127.0.0.1:7860/
# - Add and image and click Submit to get the predicted water level
import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from model import BasicCnnRegression

# Load the pre-trained model
model = BasicCnnRegression()
model.load_state_dict(torch.load('model.pth', weights_only=True))
model.eval()

# Define a function to preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def run_inference(image):
    input = preprocess_image(image)
    with torch.no_grad():
        output = model(input.unsqueeze(0))
    return output.item()

# Define the Gradio app
demo = gr.Interface(
    fn=run_inference,
    inputs=gr.Image(type="pil"),
    outputs=gr.Number(label="Output"),
    title="Image Regression App",
    description="Upload an image to get a scalar output"
)

# Launch the app
demo.launch()