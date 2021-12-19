from typing import Callable, Generic, Iterator, Optional, Tuple, TypeVar
import torch
from ._protocols import BatchSampler
from .sampler import TensorBatchSampler

T = TypeVar("T")


def _default_collate_fn(*x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    return x


class TensorDataLoader(Generic[T]):
    """
    Fast data loader for tabular data represented as tensors. This data loader drastically
    improves performance compared to PyTorch's built-in data loader by directly cutting batches
    from the data instead of getting individual items and stacking them.

    Note:
        This data loader does not provide options for multiprocessing since cutting from tensors is
        faster than sending tensors over queues.

    Attention:
        If you are using this data loader in a multi-process environment, you may need to
        explicitly pass a sampler. See :class:`~lightkit.data.DistributedTensorBatchSampler` and
        :class:`~lightkit.data.UnrepeatedDistributedTensorBatchSampler`
    """

    def __init__(
        self,
        *tensors: torch.Tensor,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        batch_sampler: Optional[BatchSampler] = None,
        collate_fn: Callable[..., T] = _default_collate_fn,
    ):
        """
        Args:
            tensors: One or more tensors of shape ``[num_datapoints, *]``. For each index, this
                dataset returns all tensors' values at that index as tuples.
            batch_size: The batch size to use. Ignored if ``batch_sampler`` is provided. If set to
                ``None``, each batch returns the full data.
            shuffle: Whether to shuffle indices that are sampled. Ignored if ``batch_sampler`` is
                provided.
            drop_last: Whether to ignore the last batch if the dataset is not divisible by the
                batch size. Ignored if ``batch_sampler`` is provided.
            batch_sampler: A batch sampler which provides either slices or batches of indices to
                gather from the dataset. By default, it initializes a
                :class:`~lightkit.data.TensorBatchSampler`.
            collate_fn: A collation function which transforms a batch of items. It receives the
                batches of each individual tensor as individual parameters.
        """
        assert len(tensors) > 0, "At least one tensor must be provided."
        assert all(
            t.size(0) == tensors[0].size(0) for t in tensors
        ), "All tensors must provide the same number of items."

        self.tensors = tensors
        self.batch_sampler = batch_sampler or TensorBatchSampler(
            self.tensors[0].size(0),
            batch_size or self.tensors[0].size(0),
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.collate_fn = collate_fn

    def __len__(self) -> int:
        return len(self.batch_sampler)  # type: ignore

    def __iter__(self) -> Iterator[T]:
        for indices in self.batch_sampler:
            item = tuple(tensor[indices] for tensor in self.tensors)
            yield self.collate_fn(*item)
