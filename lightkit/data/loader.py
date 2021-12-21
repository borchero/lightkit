from typing import Any, Iterator, TypeVar
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, TensorDataset
from .collation import collate_tuple
from .sampler import RangeBatchSampler

T_co = TypeVar("T_co", covariant=True)


class DataLoader(TorchDataLoader[T_co]):
    """
    Extension for PyTorch's builtin dataloader. This implementation allows to retrieve contiguous
    indices from a :class:`~torch.utils.data.TensorDataset` orders of magnitude faster. The
    data loader, thus, enables to implement traditional machine learning methods that exhibit a
    speed similar to the implementations found in Scikit-learn.

    Note:
        Retrieving contiguous indices is only possible when all of the following conditions apply:

        - ``shuffle=False`` or ``batch_sampler`` is of type
          :class:`~lightkit.data.RangeBatchSampler`
        - ``sampler is None``
        - ``num_workers=0``
        - ``dataset`` is not iterable
    """

    def __init__(self, dataset: Dataset[T_co], **kwargs: Any):
        """
        Args:
            dataset: The dataset from which to load the data.
            kwargs: Keyword arguments passed to :meth:`torch.utils.data.DataLoader.__init__`.
        """
        if (
            not kwargs.get("shuffle", False)
            and "sampler" not in kwargs
            and "batch_sampler" not in kwargs
            and kwargs.get("num_workers", 0) == 0
            and isinstance(dataset, TensorDataset)
        ):
            kwargs["batch_sampler"] = RangeBatchSampler(
                len(dataset),
                batch_size=kwargs.get("batch_size", 1),
                drop_last=kwargs.get("drop_last", False),
            )
            kwargs.pop("batch_size", None)
            kwargs.pop("shuffle", None)
            kwargs.pop("drop_last", None)
            kwargs.setdefault("collate_fn", collate_tuple)

        super().__init__(dataset, **kwargs)  # type: ignore

    def __iter__(self) -> Iterator[Any]:  # pylint: disable=inconsistent-return-statements
        if not (isinstance(self.dataset, TensorDataset) and self.num_workers == 0):
            return super().__iter__()

        for indices in self.batch_sampler:
            if isinstance(indices, range):
                subscript = slice(indices.start, indices.stop)
                yield self.collate_fn(tuple(t[subscript] for t in self.dataset.tensors))
            else:
                yield self.collate_fn(tuple(t[indices] for t in self.dataset.tensors))
