import torch
from torch import Tensor

from constants import EPS


class _Loss(torch.nn.Module):

    def __init__(self) -> None:
        super(_Loss, self).__init__()


class CategoricalCrossEntropy(_Loss):

    def __init__(self) -> None:
        super(CategoricalCrossEntropy, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        dims = list(input.size())
        res = torch.tensor(0.)
        eps = torch.tensor(EPS)
        div = torch.tensor(1. * dims[0])

        for i in range(dims[0]):
            for x in range(dims[1]):
                res = res - input[i][x] * torch.log(target[i][x] + eps)

        return torch.stack([res / div])
