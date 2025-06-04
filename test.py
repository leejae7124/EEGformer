import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU 개수:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())  # 여기서 터질 수도 있음
