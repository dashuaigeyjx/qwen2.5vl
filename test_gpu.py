import torch
print("CUDA 可用:", torch.cuda.is_available())
print("GPU 数量:", torch.cuda.device_count())
print("当前 GPU 名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")
