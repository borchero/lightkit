from typing import Tuple
import torch


def collate_tuple(*batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Collate a tuple of batch items by returning the input tuple. This is the default used by
    :class:`~lightkit.data.TensorDataLoader`.
    """
    return batch


def collate_tensor(*batch: torch.Tensor) -> torch.Tensor:
    """
    Collates a tuple of batch items into the first tensor. Might be useful if only a single tensor
    is passed to :class:`~lightkit.data.TensorDataLoader`.
    """
    return batch[0]
