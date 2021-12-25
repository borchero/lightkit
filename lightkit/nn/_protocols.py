# pylint: disable=missing-class-docstring,missing-function-docstring
import sys
from os import PathLike
from typing import Generic, Iterator, OrderedDict, Protocol, Tuple, Type, TypeVar, Union
import torch
from torch import nn

C = TypeVar("C", covariant=True)
M = TypeVar("M", bound="ConfigurableModule")  # type: ignore

if sys.version_info < (3, 9, 0):
    # PathLike is not generic for Python 3.9
    PathType = Union[str, PathLike]
else:
    PathType = Union[str, PathLike[str]]


class AnyConfigurableModule(Protocol):
    @classmethod
    def load(cls: Type[M], path: PathType) -> M:
        ...

    def save(self, path: PathType, compile_model: bool = False) -> None:
        ...

    def save_config(self, path: PathType) -> None:
        ...

    def named_children(self) -> Iterator[Tuple[str, nn.Module]]:
        ...

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        ...

    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor]) -> None:
        ...


class ConfigurableModule(AnyConfigurableModule, Protocol, Generic[C]):
    @property
    def config(self) -> C:
        ...
