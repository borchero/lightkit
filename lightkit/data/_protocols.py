# pylint: disable=missing-function-docstring
from typing import Iterator, Protocol, Union
import torch


class BatchSampler(Protocol):
    """
    Simple protocol defining the requirements for a batch sampler for tensors.
    """

    @property
    def batch_size(self) -> int:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[Union[slice, torch.Tensor]]:
        ...
