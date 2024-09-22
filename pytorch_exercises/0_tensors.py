import torch
from torchvision.io import read_image
from utils import Timer
from typing import Tuple

def sample_tensor_image(image_path: str ='datasets/rabbits_2024_08_12_25_15sec/images/001.png') -> torch.tensor:
  # Load the contents of the image as a tensor
  image_tensor = read_image(image_path)
  # Make float and normalize
  image_tensor = image_tensor.float() / 255.0
  return image_tensor

def sample_tensor_random(sizes: list = [2000, 3000, 4]) -> torch.tensor:
  # Create a tensor of the provided size (deafults to 2000x3000x4) filled with random values
  x = torch.rand(sizes)
  return x

def perform_some_operations(x: torch.tensor, x_mul: torch.tensor, device: str, timer: Timer) -> torch.tensor:
  # Make sure the input tensors are in the right device
  x = x.to(device)
  x_mul = x_mul.to(device)

  timer.start()

  # Create a tensor with 100 of those images, along a fourth new dimension (batch), on the CPU
  x_big = torch.stack([x] * 35)  # Fun fact, this is the most images I can add before GPU OoO on my nVidia GTXi 1650

  # Multiply it multiple times
  for i in range(10):
    #x_big = torch.matmul(x_big, x_mul) / 3072.0
    x_big = torch.matmul(x_big, x_mul)

  timer.stop()

  return x_big

def peek_at_tensor(x: torch.tensor, name: str = 'x'):
  print(f"{name}:")
  print(f"\tshape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
  print(f"\tmin: {x.min()}, max: {x.max()}, mean: {x.mean()}, std: {x.std()}")

t_cpu = Timer()
t_gpu = Timer()

# Play with torch defaults
# torch.set_default_device('cuda')
# torch.set_default_dtype(torch.float16)

# Create a sample tensor from an image
x = sample_tensor_image()
x = torch.mean(x, dim=0)  # Reduces the tensor to a single channel, calculating the mean over the others
peek_at_tensor(x, 'x initially')
# Also create a random square tensor to use as a multiplier
x_mul = sample_tensor_random([3072, 3072])
x_mul = x_mul.float() / x_mul.max()  # Normalize the multiplier
peek_at_tensor(x_mul, 'x_mul initially')

# Repeat on GPU
x_gpu = perform_some_operations(x, x_mul, 'cuda', t_gpu)
print(f"GPU timing: {t_gpu}")

# Test operations on CPU
x_cpu = perform_some_operations(x, x_mul, 'cpu', t_cpu)
print(f"CPU timing: {t_cpu}")

# Print some of the contents
peek_at_tensor(x_gpu, "x_gpu")
peek_at_tensor(x_cpu, "x_cpu")
