import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torchvision import models


def preprocess(image_path: str, input_size: tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Preprocesses an image for PyTorch model inference.

    Args:
        image_path: Path to the input image.
        input_size: The target size (height, width) for the model input.

    Returns:
        A preprocessed PyTorch tensor ready for inference.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(input_size)
    img_data = np.array(img).astype("float32")

    # Normalize to [0, 1]
    img_data /= 255.0

    # Transpose from (H, W, C) to (C, H, W)
    img_data = np.transpose(img_data, (2, 0, 1))

    # Convert to PyTorch tensor and add batch dimension
    input_tensor = torch.from_numpy(img_data).unsqueeze(0)

    return input_tensor


def main():
    """
    Main function to load the model and run inference on sample images.
    """

    # Model path - using safetensors format
    # If you have a .safetensors file, specify it here
    # Otherwise, we'll use pretrained weights from torchvision
    safetensor_path = "../../models/weights/resnet-152.safetensors"
    image_dir = "."
    sample_images = ["cat.jpg", "dog.jpg"]

    # Key Imagenet-1k classes:
    # - 285: Egyptian Cat
    # - 226: Briard
    # Ref: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

    # --- Check GPU availability ---
    if not torch.cuda.is_available():
        print("Error: CUDA is not available.")
        print("Please ensure you have a CUDA-compatible GPU and PyTorch with CUDA support installed.")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        return

    device = torch.device("cuda")
    print("✓ CUDA is available")
    print(f"✓ Using device: {device}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ PyTorch version: {torch.__version__}\n")

    # --- 1. Load the PyTorch model ---
    print("Loading ResNet-152 model...")

    # Create the model architecture
    model = models.resnet152(weights=None)  # Start with no weights

    # Try to load from safetensors if available
    if os.path.exists(safetensor_path):
        print(f"Loading weights from {safetensor_path}...")
        state_dict = load_file(safetensor_path)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Safetensor file not found at {safetensor_path}")
        print("Loading pretrained ImageNet weights from torchvision...")
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)

    # Move model to GPU and set to evaluation mode
    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully!")
    print("✓ Model moved to GPU")
    print("✓ Model set to evaluation mode\n")

    # --- 2. Process each sample image ---
    for image_file in sample_images:
        image_path = os.path.join(image_dir, image_file)
        if not os.path.exists(image_path):
            print(f"\nWarning: Sample image not found at {image_path}. Skipping.")
            continue

        print(f"\n--- Processing {image_file} ---")

        # --- 3. Preprocess the image ---
        input_tensor = preprocess(image_path)
        input_tensor = input_tensor.to(device)  # Move input to GPU

        # --- 4. Run inference ---
        print("Running inference 100 times...")
        total_duration = 0.0
        last_output = None

        with torch.no_grad():  # Disable gradient computation for inference
            for _ in range(100):
                start = time.time()
                output = model(input_tensor)
                torch.cuda.synchronize()  # Wait for GPU computation to finish
                elapsed = time.time() - start
                total_duration += elapsed
                last_output = output

        print(f"Inference completed 100 times in {total_duration:.4f}s")
        print(f"Average time per inference: {(total_duration / 100) * 1000:.2f}ms")

        # --- 5. Post-process the result ---
        if last_output is not None:
            probabilities = F.softmax(last_output, dim=1)[0]
            probabilities_np = probabilities.cpu().numpy()
            predicted_class_index = int(torch.argmax(probabilities).item())
            confidence = probabilities_np[predicted_class_index]

            print(f"Prediction: '{predicted_class_index}' with {confidence:.2%} confidence.")


if __name__ == "__main__":
    main()
