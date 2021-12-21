# pylint: disable=missing-class-docstring,missing-function-docstring
from lightkit.data import RangeBatchSampler


def test_tensor_batch_sampler():
    sampler = RangeBatchSampler(5, 2)
    assert list(sampler) == [range(0, 2), range(2, 4), range(4, 6)]

    sampler = RangeBatchSampler(5, 2, drop_last=True)
    assert list(sampler) == [range(0, 2), range(2, 4)]
