import math
from typing import Iterator, List, Union
import torch
from torch.utils.data import Sampler


class TensorBatchSampler(Sampler[Union[slice, torch.Tensor]]):
    """
    Sampler providing batches of (contiguous) indices. These indices can be used for constructing
    batches swiftly.

    Attention:
        This sampler should only be used within a single process. Use
        :class:`DistributedTensorBatchSampler` or :class:`UnrepeatedDistributedTensorBatchSampler`
        otherwise.
    """

    def __init__(
        self, num_items: int, batch_size: int, shuffle: bool = False, drop_last: bool = False
    ):
        """
        Args:
            num_items: The number of items to sample from.
            batch_size: The number of items to sample for each batch.
            shuffle: Whether to shuffle the indices during sampling. Note that sampling increases
                the complexity of the sampler significantly as all indices need to be sampled and
                non-contiguous slices need to be cut from the input tensors. Nonetheless, this
                won't be noticeable for most practical applications.
            drop_last: Whether to drop the last batch if ``num_items`` is not divisible by
                ``batch_size``.
        """
        super().__init__(None)
        self.dataset_size = num_items
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self) -> Iterator[Union[slice, torch.Tensor]]:
        if self.shuffle:
            indices = torch.randperm(self.dataset_size)
        for i in range(len(self)):
            sample = slice(i * self.batch_size, (i + 1) * self.batch_size)
            if self.shuffle:
                yield indices[sample]  # type: ignore
            else:
                yield sample


class DistributedTensorBatchSampler(Sampler[torch.Tensor]):
    """
    Sampler providing batches of sampled indices in a distributed environment. If the data is not
    divisible by the number of processes, this sampler yields randomly selected duplicate items.
    This way, it can be ensured that every process runs equally many batches. This sampler
    therefore always shuffles the data.

    Note:
        Typically, you do not need to use this sampler directly. If you are using
        :class:`~lightkit.BaseEstimator`, its :meth:`~lightkit.BaseEstimator.trainer` method will
        provide a trainer that automatically uses this sampler if required.
    """

    def __init__(
        self,
        num_items: int,
        batch_size: int,
        num_replicas: int,
        rank: int,
        drop_last: bool = False,
        seed: int = 0,
    ):
        """
        Args:
            num_items: The number of items to sample from.
            batch_size: The number of items to sample per process for each batch.
            num_replicas: The total number of processes for which the sampler is used (i.e. the
                world size).
            rank: The rank of the process for which this sampler is providing items.
            drop_last: If set to ``True``, it ensures that no duplicate items are yielded when
                iterating over the dataset. If set to ``False`` and ``num_items`` is not divisible
                by ``batch_size * num_replicas``, duplicate items some duplicate indices will be
                chosen at random such that every process receives equally many items.
            seed: The seed to use for sampling indices.
        """
        super().__init__(None)

        self.dataset_size = num_items

        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last

        self.seed = seed
        self.current_epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // (self.num_replicas * self.batch_size)
        return math.ceil(self.dataset_size / (self.num_replicas * self.batch_size))

    def __iter__(self) -> Iterator[torch.Tensor]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.current_epoch)

        permutation = torch.randperm(self.dataset_size, generator=g)
        chunk_size = math.ceil(self.dataset_size / self.num_replicas)
        local_choices = permutation[chunk_size * self.rank : chunk_size * (self.rank + 1)]
        if self.drop_last:
            local_choices = local_choices[: len(self) * self.batch_size]
        elif len(local_choices) != chunk_size:
            # This happens if the data is not divisible by the number of replicas. We need to
            # choose a random element in some processes. We just do this by picking the first
            # elements of the permutation.
            num_unfulfilled = chunk_size * self.num_replicas - self.dataset_size
            num_fulfilled = self.num_replicas - num_unfulfilled
            idx = self.rank - num_fulfilled
            local_choices = torch.cat([local_choices, permutation[idx : idx + 1]])

        for i in range(len(self)):
            yield local_choices[i * self.batch_size : (i + 1) * self.batch_size]

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler to ensure that different indices are sampled in each epoch.
        """
        self.current_epoch = epoch


class UnrepeatedDistributedTensorBatchSampler(Sampler[slice]):
    """
    Sampler providing contiguous indices with possibly unevenly sized batches. This class is
    similar to :class:`DistributedTensorBatchSampler`. However, it yields items in-order and does
    not return duplicate items, possibly yielding unequally many batches for different parallel
    replicas. Thus, this should only be used for testing where it can be ensured that each replica
    receives at least one batch. See also
    :class:`pytorch_lightning.overrides.distributed.UnrepeatedDistributedSampler`.

    Note:
        Typically, you do not need to use this sampler directly. If you are using
        :class:`~lightkit.BaseEstimator`, its :meth:`~lightkit.BaseEstimator.trainer` method will
        provide a trainer that automatically uses this sampler if required.
    """

    def __init__(self, num_items: int, batch_size: int, num_replicas: int, rank: int):
        """
        Args:
            num_items: The number of items to sample from.
            batch_size: The number of items to sample for each batch.
            num_replicas: The total number of processes for which the sampler is used (i.e. the
                world size).
            rank: The rank of the process for which this sampler is providing items.
        """
        super().__init__(None)

        self.dataset_size = num_items

        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank

    def __len__(self) -> int:
        # This only applies to rank 0, but that is fine for displaying a progress bar
        return math.ceil(math.ceil(self.dataset_size / self.num_replicas) / self.batch_size)

    def __iter__(self) -> Iterator[slice]:
        chunk_sizes = [
            self.dataset_size // self.num_replicas + int(i < self.dataset_size % self.num_replicas)
            for i in range(self.num_replicas)
        ]
        prev_size = sum(chunk_sizes[: self.rank])
        local_size = chunk_sizes[self.rank]

        for i in range(math.ceil(local_size / self.batch_size)):
            yield slice(
                prev_size + i * self.batch_size,
                min(prev_size + (i + 1) * self.batch_size, prev_size + local_size),
            )


class IndexTensorBatchSamplerWrapper:
    """
    Wrapper around a sampler providing batches as tensors (or slices) and tracking the indices.
    """

    def __init__(
        self,
        batch_sampler: Union[
            TensorBatchSampler,
            DistributedTensorBatchSampler,
            UnrepeatedDistributedTensorBatchSampler,
        ],
    ):
        self.batch_sampler = batch_sampler
        self.seen_batch_indices: List[List[int]] = []

    @property
    def batch_size(self) -> int:
        """
        Returns the batch size of the underlying sampler.
        """
        return self.batch_sampler.batch_size

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __iter__(self) -> Iterator[Union[slice, torch.Tensor]]:
        self.seen_batch_indices = []
        for batch in self.batch_sampler:
            if isinstance(batch, torch.Tensor):
                self.seen_batch_indices.append(batch.tolist())
            else:
                assert (
                    batch.step is None
                ), "Batch sampler must not provide slices with step size other than 1."
                self.seen_batch_indices.append(list(range(batch.start, batch.stop)))
            yield batch
