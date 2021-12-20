from typing import Callable, cast, Optional, TypeVar, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .collation import collate_tuple
from .loader import TensorDataLoader
from .types import TensorData

T = TypeVar("T")


def data_loader_from_tensor_data(
    *data: Union[TensorData, TensorDataset],
    batch_size: Optional[int] = None,
    shuffle: bool = False,
    drop_last: bool = False,
    collate_fn: Callable[..., T] = collate_tuple,
) -> DataLoader[T]:
    """
    Returns a data loader for tabular data.

    Args:
        data: The tensors for which to create a data loader. If a :class:`TensorDataset` is
            provided, only a single item must be passed. Any :class:`Dataset` instances that are
            passed are loaded into memory.
        batch_size: The batch size to use. If set to ``None``, each batch returns the full data.
        shuffle: Whether to shuffle indices that are sampled.
        drop_last: Whether to ignore the last batch if the dataset is not divisible by the batch
            size.
        collate_fn: A function which converts a tuple of batches into a custom type. By default,
            the tuple is simply returned.

    Returns:
        A data loader providing the items from the passed tensors.

    Attention:
        The returned data loader is *not* an instance of :class:`~torch.utils.data.DataLoader`, but
        an instance of :class:`TensorDataLoader`. However, it provides many of the same features
        and can be used in almost all places where PyTorch's native data loader is used. Thus, we
        cast the type for the user's convenience.
    """
    assert len(data) > 0, "At least one tensor must be provided."
    assert (
        not isinstance(data[0], TensorDataset) or len(data) == 1
    ), "If a TensorDataset is passed, no more than one item must be provided."

    if isinstance(data[0], TensorDataset):
        tensors = data[0].tensors
    else:
        tensors = tuple(_to_tensor(cast(TensorData, t)) for t in data)

    dl = TensorDataLoader(
        *tensors,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    return cast(DataLoader[T], dl)


def _to_tensor(data: TensorData) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        return data
    return torch.stack([item for item in data])  # pylint: disable=unnecessary-comprehension
