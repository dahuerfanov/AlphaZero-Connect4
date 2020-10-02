# for TPU
import torch_xla.core.xla_model as xm

from AlphaZero import policyIter

device = device = xm.xla_device()
print("Using TPU!")
policyIter(name_cnn_model="cnn_v128_model", device=device)
