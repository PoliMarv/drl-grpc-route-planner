import ray
import tensorflow as tf
import torch

print("Ray version:", ray.__version__)
print("TensorFlow GPU available:", tf.test.is_gpu_available())
print("PyTorch GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("PyTorch GPU name:", torch.cuda.get_device_name(0))
