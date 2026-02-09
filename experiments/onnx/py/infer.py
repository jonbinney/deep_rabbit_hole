import os
import time

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
    model_path = "../../../models/onnx/resnet-152.onnx"
    image_dir = ".."
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

    # --- Configure session with CUDA (GPU) execution provider ONLY ---
    # Disable fallback to CPU by only specifying CUDA provider
    # If CUDA is not available, the session creation will fail instead of falling back to CPU
    providers = ["CUDAExecutionProvider"]

    # Check if CUDA is available
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available_providers:
        print("Error: CUDA execution provider not available.")
        print(f"Available providers: {available_providers}")
        print("Please ensure CUDA/GPU is available and onnxruntime-gpu is installed.")
        return

    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    print("✓ Model loaded successfully!")
    print("✓ CUDA execution provider configured (GPU acceleration enabled)")
    print("✓ CPU fallback is DISABLED - will fail if GPU is not available\n")

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
        print("Running inference 100 times...")

        total_duration = 0.0
        last_output = None

        for _ in range(100):
            start = time.time()
            result = session.run(None, {input_name: input_tensor})
            elapsed = time.time() - start
            total_duration += elapsed
            last_output = result[0]

        print(f"Inference completed 100 times in {total_duration:.4f}s")
        print(f"Average time per inference: {(total_duration / 100) * 1000:.2f}ms")

        # --- 5. Post-process the result ---
        probabilities = softmax(last_output)[0]
        predicted_class_index = np.argmax(probabilities)
        confidence = probabilities[predicted_class_index]

        print(f"Prediction: '{predicted_class_index}' with {confidence:.2%} confidence.")


if __name__ == "__main__":
    main()
