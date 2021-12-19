# pylint: disable=missing-class-docstring,missing-function-docstring
from typing import cast, List
import torch
from lightkit.data import TensorBatchSampler


def test_tensor_batch_sampler():
    sampler = TensorBatchSampler(5, 2)
    assert list(sampler) == [slice(0, 2), slice(2, 4), slice(4, 6)]

    sampler = TensorBatchSampler(5, 2, drop_last=True)
    assert list(sampler) == [slice(0, 2), slice(2, 4)]

    sampler = TensorBatchSampler(5, 2, shuffle=True)
    assert sorted(torch.cat(cast(List[torch.Tensor], list(sampler))).tolist()) == [0, 1, 2, 3, 4]
