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
import cv2

def load_model(model_path):
    checkpoint = torch.load(model_path, weights_only=False)
    model_args = checkpoint['model_args']
    model = BasicCnnRegression(**model_args)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return (model, model_args)

def run_inference(model, input, transforms = None):
    if transforms is not None:
        input = transforms(input)
    with torch.no_grad():
        output = model(input.unsqueeze(0))
    return output.item()

def run_gradio_app(model, crop_box = None):
    transforms = get_transforms(crop_box)

    # Define the Gradio app
    demo = gr.Interface(
        fn=lambda image: run_inference(model, image, transforms),
        inputs=gr.Image(type="pil"),
        outputs=gr.Number(label="Output"),
        title="Image Regression App",
        description="Upload an image to get a scalar output"
    )

    # Launch the app
    demo.launch()

def run_dataset_inference(model, dataset_dir, annotations_file, normalized_output, crop_box = None, **kwargs):

    # Load the dataset
    dataset = get_data_loader(dataset_dir + '/images', dataset_dir + '/annotations/' + annotations_file, shuffle=False, crop_box=crop_box, normalize_output=normalized_output)

    # Run inference
    loss = 0
    n_images = 0
    for i, (images, depths, filenames) in enumerate(dataset):
        mse = 0
        for image, depth, filename in zip(images, depths, filenames):
            output = run_inference(model, image)
            error = abs(output - depth.item())

            # use OpenCV to display the image and wait for a key
            img = cv2.imread(filename)
            # Convert to grayscale and normalize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
            # Calculate mean and standard deviation
            (means, stds) = cv2.meanStdDev(img)
 
            print(f"Filename: {filename}, Infered: {output:.2f}, Actual: {depth.item():.2f}, Error: {error:.2f}, mean: {means[0][0]:.2f}, std: {stds[0][0]:.2f}")
            mse += error**2
        mse /= len(images)
        print(f"MSE ({i}): {mse}")
        loss += mse
        n_images += len(images)

    print(f"Average loss: {loss / len(dataset)}, images: {n_images}, dataset size: {len(dataset)}")

if __name__ == "__main__":
    # Parse program arguments
    parser = argparse.ArgumentParser(description='Deep Water Level')
    parser.add_argument('-m', '--model_path', type=str, default='model.pth', help='Path to the model file')
    parser.add_argument('--normalized_output', type=bool, default=False, help='Set to true if the model was trained with depth values normalized to [-1, 1] range')
    parser.add_argument('--crop_box', nargs=4, type=int, default=None, help='Box with which to crop images, of form: top left height width')

    # If these arguments are provided, then the model will be run against the dataset, showing results.
    parser.add_argument('--dataset_dir', type=str, default='datasets/water_test_set5', help='Path to the dataset directory')
    parser.add_argument('--annotations_file', type=str, default='filtered.csv', help='File name of the JSON file containing annotations')

    args = parser.parse_args()

    (model, model_args) = load_model(args.model_path)

    # Load the model
    if 'dataset_dir' in args and 'annotations_file' in args:
        run_dataset_inference(model, **vars(args))
    else:
        run_gradio_app(model, model_args['crop_box'])