# This script is useful to test the deep water level model.
# It can be used to test the model on a single image manually, or on a dataset as a batch.
#
# If the dataset parameters (dataset_dir and annotations_file) are provided, the model will be run on the dataset,
# printing the inferred values vs actuals and the average loss.
#
# If the dataset parameters are omitted, the script will run
# a Gradio interface to run the deep water level model against manually uploaded images.
#
# In order to run it like this:
# - Run training or get a model, save it as model.pth at the root of the project
# - Run this script, then open the Gradio app at http://127.0.0.1:7860/
# - Add and image and click Submit to get the predicted water level
import gradio as gr
import torch
from PIL import Image
from model import BasicCnnRegression
from data import get_transforms, get_data_loader
import argparse

def load_model(model_path):
    # Load the pre-trained model
    model = BasicCnnRegression()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

# Define a function to preprocess the image
def preprocess_image(image):
    transforms = get_transforms() 
    return transforms(image)

def run_inference(model, image):
    input = preprocess_image(image)
    with torch.no_grad():
        output = model(input.unsqueeze(0))
    return output.item()

def run_gradio_app(model):
    # Define the Gradio app
    demo = gr.Interface(
        fn=lambda image: run_inference(model, image),
        inputs=gr.Image(type="pil"),
        outputs=gr.Number(label="Output"),
        title="Image Regression App",
        description="Upload an image to get a scalar output"
    )

    # Launch the app
    demo.launch()

def run_dataset_inference(model, dataset_dir, annotations_file):

    # Load the dataset
    dataset = get_data_loader(dataset_dir + '/images', dataset_dir + '/annotations/' + annotations_file, shuffle=False)

    # Run inference
    loss = 0
    n_images = 0
    for i, (images, depths, filenames) in enumerate(dataset):
        mse = 0
        for image, depth, filename in zip(images, depths, filenames):
            output = run_inference(model, image)
            error = abs(output - depth.item())
            print(f"Filename: {filename}, Infered: {output:.2f}, Actual: {depth.item():.2f}, Error: {error:.2f}")
            mse += error**2
        mse /= len(images)
        print(f"MSE ({i}): {mse}")
        loss += mse
        n_images += len(images)

    print(f"Average loss: {loss / len(dataset)}, images: {n_images}, dataset size: {len(dataset)}")

if __name__ == "__main__":
    # Parse program arguments
    parser = argparse.ArgumentParser(description='Deep Water Level')
    parser.add_argument('-m', '--model_path', type=str, default='model1.pth', help='Path to the model file')

    # If these arguments are provided, then the model will be run against the dataset, showing results.
    parser.add_argument('--dataset_dir', type=str, default='datasets/water_2024_11_01_set2', help='Path to the dataset directory')
    parser.add_argument('--annotations_file', type=str, default='manual_annotations.json', help='File name of the JSON file containing annotations')

    args = parser.parse_args()

    model = load_model(args.model_path)

    # Load the model
    if 'dataset_dir' in args and 'annotations_file' in args:
        run_dataset_inference(model, args.dataset_dir, args.annotations_file)
    else:
        run_gradio_app(model)