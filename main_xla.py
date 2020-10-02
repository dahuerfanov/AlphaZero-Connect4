from AlphaZero import policyIter
import torch
# for TPU
import torch_xla
import torch_xla.core.xla_model as xm


device = device = xm.xla_device()
print("Using TPU!")
policyIter(name_cnn_model="cnn_v128_model", device=device)
