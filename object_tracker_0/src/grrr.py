from groundingdino.util.inference import predict,load_model
import groundingdino.datasets.transforms as T
from PIL import Image
from huggingface_hub import hf_hub_download

# Use this command for evaluate the Grounding DINO model
model_config_path = "code/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
repo_id = "ShilongLiu/GroundingDINO"
filename = "groundingdino_swint_ogc.pth"

model = load_model("code/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

init_image = Image.open("/home/julian/aaae/deep-rabbit-hole/datasets/rabbits_2024_08_12_25_15sec/images/001.png")
init_image = init_image.convert("RGB")

transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image_tensor, _ = transform(init_image, None) # 3, h, w

boxes, logits, phrases = predict(model, image_tensor, "rabbit", 0.15, 0.15, device='cpu')
print(f"boxes: {boxes}")
