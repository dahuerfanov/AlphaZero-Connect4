import torch

from AlphaZero import policyIter

# for GPU
if torch.cuda.is_available():
    device = torch.cuda.device(0)
    print("Using GPU!", torch.cuda.get_device_name(None))
else:
    device = torch.device("cpu")
    print("Using CPU :(")

policyIter(work_path="data/", name_cnn_model="cnn_v128_model", device=device)
