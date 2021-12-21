import math
from typing import Iterator
from torch.utils.data import Sampler


class RangeBatchSampler(Sampler[range]):
    """
    Sampler providing batches of contiguous indices. This sampler can be used with
    :class:`lightkit.data.DataLoader` to provide significant speedups for tensor datasets.
    """

    def __init__(self, num_items: int, batch_size: int, drop_last: bool = False):
        """
        Args:
            num_items: The number of items to sample from.
            batch_size: The number of items to sample for each batch.
            drop_last: Whether to drop the last batch if ``num_items`` is not divisible by
                ``batch_size``.
        """
        super().__init__(None)
        self.dataset_size = num_items
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self) -> Iterator[range]:
        for i in range(len(self)):
            sample = range(i * self.batch_size, (i + 1) * self.batch_size)
            yield sample
