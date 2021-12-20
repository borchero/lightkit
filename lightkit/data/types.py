from typing import Sequence, TypeVar, Union
import numpy as np
import numpy.typing as npt
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from .loader import TensorDataLoader

T = TypeVar("T")

TensorData = Union[
    npt.NDArray[np.float32],
    torch.Tensor,
    TorchDataset[torch.Tensor],
]
TensorData.__doc__ = """
Type annotation for functions accepting any kind of tensor data as input. Consider using this
annotation if your methods in an estimator derived from :class:`lightkit.BaseEstimator` work on
tensors.
"""

DataLoader = Union[
    LightningDataModule,
    TorchDataLoader[T],
    Sequence[TorchDataLoader[T]],
    TensorDataLoader[T],
    Sequence[TensorDataLoader[T]],
]
DataLoader.__doc__ = """
Generic type annotation for functions accepting any data loader as input. Consider using this
annotation for the implementation of methods in an estimator derived from
:class:`lightkit.BaseEstimator`.
"""
