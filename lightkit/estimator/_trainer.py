# pylint: disable=missing-class-docstring
from typing import Any, cast, Dict, Optional, TypeVar, Union
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import DataLoader
from lightkit.data import TensorDataLoader
from lightkit.data.sampler import (
    DistributedTensorBatchSampler,
    IndexTensorBatchSamplerWrapper,
    UnrepeatedDistributedTensorBatchSampler,
)

T = TypeVar("T")


class LightkitTrainer(pl.Trainer):
    def prepare_dataloader(
        self, dataloader: Any, shuffle: bool, mode: Optional[RunningStage] = None
    ) -> Any:
        # Call super first to recursively call this function for collections of data loaders. It
        # does not apply any modification for the `TensorDataLoader` as it does not subclass
        # PyTorch's data loader.
        result = super().prepare_dataloader(dataloader, shuffle, mode)

        if isinstance(result, TensorDataLoader) and self._requires_distributed_sampler(result):
            sampler_cls = (
                UnrepeatedDistributedTensorBatchSampler
                if mode == RunningStage.PREDICTING
                else DistributedTensorBatchSampler
            )
            kwargs = cast(Dict[str, int], self.distributed_sampler_kwargs)
            sampler = sampler_cls(
                num_items=result.tensors[0].size(0),
                batch_size=result.batch_sampler.batch_size,
                drop_last=result.batch_sampler.drop_last,
                num_replicas=kwargs["num_replicas"],
                rank=kwargs["rank"],
            )
            if mode == RunningStage.PREDICTING:
                sampler = IndexTensorBatchSamplerWrapper(sampler)
            result = TensorDataLoader(
                *result.tensors, batch_sampler=sampler, collate_fn=result.collate_fn
            )

        return result

    def _requires_distributed_sampler(
        self, dataloader: Union[DataLoader[T], TensorDataLoader[T]]
    ) -> bool:
        if isinstance(dataloader, TensorDataLoader):
            return (
                self._accelerator_connector.replace_sampler_ddp
                and self._accelerator_connector.is_distributed
                and not isinstance(
                    dataloader.batch_sampler,
                    (DistributedTensorBatchSampler, UnrepeatedDistributedTensorBatchSampler),
                )
            )
        return super()._requires_distributed_sampler(dataloader)
