import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:

    max_values = x.max(dim=dim, keepdim=True).values
    shift_values = x - max_values
    exp_values = torch.exp(shift_values)
    exp_sum_values = exp_values.sum(dim=dim, keepdim=True)
    return exp_values / exp_sum_values