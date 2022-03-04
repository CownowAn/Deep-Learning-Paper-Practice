from signal import default_int_handler
from numpy import isin
import torch

from collections import OrderedDict
from torchmeta.modules import MetaModule

def compute_accurcy(logits, targets):
    """
    Compute the accuracy

    Args:
        logits : predicted logits
        targets : aiming classes

    Returns:
        accuracy : accuracy between prediction and target
    """
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()

def tensors_to_device(tensors, device=torch.device('cpu')):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device) for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(key, tensors_to_device(tensor, device=device)) for (key, tensor) in tensors.items()])
    else:
        raise NotImplementedError()

class ToTensor1D(object):
    """
    Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.
    
    Converts automatically the array to `float32`.
    """
    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'