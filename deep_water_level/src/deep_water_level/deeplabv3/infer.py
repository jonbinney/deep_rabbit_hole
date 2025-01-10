import torch
from torchvision import models
from torchvision.transforms import functional as F
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from deep_water_level.deeplabv3.utils import preprocess_image

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


def visualize_segmentation(image, mask):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="jet", alpha=0.7)
    plt.title("Segmentation Mask")
    plt.axis("off")

    combined = image.clone()
    combined[pred == 1] = torch.tensor([1.0, 0, 0])

    plt.subplot(1, 3, 3)
    plt.imshow(combined)
    plt.title("Image with Segmentation Mask")
    plt.axis("off")
    plt.show()


model = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier = DeepLabHead(2048, num_classes=2)

model.load_state_dict(torch.load("deeplabv3_pool.pth", weights_only=False))
model.eval()

image_path = "/Users/amarcu/code/deep_rabbit_hole/datasets/water_test_set5/images/2024-10-29/0-1730238001.jpg"

image = Image.open(image_path)
preprocessed_image = preprocess_image(image).unsqueeze(0)

output = model(preprocessed_image)["out"]
pred = torch.argmax(output, dim=1).squeeze().numpy()

visualize_segmentation(transforms.ToTensor()(image).permute(1, 2, 0), pred)
