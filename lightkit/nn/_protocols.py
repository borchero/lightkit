import sys
from os import PathLike
from typing import Generic, OrderedDict, Protocol, Type, TypeVar, Union
import torch

C = TypeVar("C")
M = TypeVar("M", bound="ConfigurableModule")  # type: ignore

if sys.version_info < (3, 9, 0):
    # PathLike is not generic for Python 3.9
    PathType = Union[str, PathLike]
else:
    PathType = Union[str, PathLike[str]]


class ConfigurableModule(Protocol, Generic[C]):  # pylint: disable=missing-class-docstring
    # pylint: disable=missing-function-docstring

    def __init__(self, config: C):  # pylint: disable=super-init-not-called
        ...

    @classmethod
    def load(cls: Type[M], path: PathType) -> M:
        ...

    @classmethod
    def _get_config_class(cls: Type[M]) -> Type[C]:
        ...

    @property
    def config(self) -> C:
        ...

    def save(self, path: PathType, compile_model: bool = False) -> None:
        ...

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        ...

    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor]) -> None:
        ...
