import torch.nn.functional as F
import torch

def one_warm_to_hot(tensor):
    categorical = torch.argmax(tensor, dim=1) # Assume 1 is the batch dim as usually is the case
    one_hot_channel_last = F.one_hot(categorical, num_classes=tensor.size(1))
    total_dims = len(tensor.shape)
    permutation_order = [0, total_dims - 1]
    remaining = list(range(1, total_dims - 1))
    permutation_order.extend(remaining)
    return torch.permute(one_hot_channel_last, tuple(permutation_order))
