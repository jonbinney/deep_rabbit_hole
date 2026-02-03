import os

import numpy as np
import onnxruntime as ort
from PIL import Image


def preprocess(image_path: str, input_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocesses an image for ONNX model inference.

    Args:
        image_path: Path to the input image.
        input_size: The target size (height, width) for the model input.

    Returns:
        A preprocessed numpy array ready for inference.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(input_size)
    img_data = np.array(img).astype("float32")

    # Normalize to [0, 1]
    img_data /= 255.0

    # Transpose from (H, W, C) to (C, H, W)
    img_data = np.transpose(img_data, (2, 0, 1))

    # Add a batch dimension to create (N, C, H, W)
    input_tensor = np.expand_dims(img_data, axis=0)

    return input_tensor


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def main():
    """
    Main function to load the model and run inference on sample images.
    """

    # From https://huggingface.co/Xenova/resnet-152/blob/main/onnx/model.onnx
    # ResNet-152 trained on ImageNet-1k with ONNX saved in HuggingFace transformers compatible format
    model_path = "../models/onnx/resnet-152.onnx"
    image_dir = "."
    sample_images = ["cat.jpg", "dog.jpg"]

    # Key Imagenet-1k classes:
    # - 285: Egyptian Cat
    # - 226: Briard
    # Ref: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

    # --- 1. Load the ONNX model ---
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please ensure you have a trained 'model.onnx' in the 'models/' directory.")
        return

    print(f"Loading model from {model_path}...")
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    print("Model loaded successfully.")

    # --- 2. Process each sample image ---
    for image_file in sample_images:
        image_path = os.path.join(image_dir, image_file)
        if not os.path.exists(image_path):
            print(f"\nWarning: Sample image not found at {image_path}. Skipping.")
            continue

        print(f"\n--- Processing {image_file} ---")

        # --- 3. Preprocess the image ---
        input_tensor = preprocess(image_path)

        # --- 4. Run inference ---
        print("Running inference...")
        result = session.run(None, {input_name: input_tensor})
        output_tensor = result[0]

        # --- 5. Post-process the result ---
        probabilities = softmax(output_tensor)[0]
        predicted_class_index = np.argmax(probabilities)
        confidence = probabilities[predicted_class_index]

        print(f"Prediction: '{predicted_class_index}' with {confidence:.2%} confidence.")


if __name__ == "__main__":
    main()
