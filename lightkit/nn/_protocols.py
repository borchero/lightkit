from os import PathLike
from typing import Generic, OrderedDict, Protocol, Type, TypeVar
import torch

C = TypeVar("C")
M = TypeVar("M", bound="ConfigurableModule")  # type: ignore


class ConfigurableModule(Protocol, Generic[C]):  # pylint: disable=missing-class-docstring
    # pylint: disable=missing-function-docstring

    def __init__(self, config: C):  # pylint: disable=super-init-not-called
        ...

    @classmethod
    def load(cls: Type[M], path: PathLike[str]) -> M:
        ...

    @classmethod
    def _get_config_class(cls: Type[M]) -> Type[C]:
        ...

    @property
    def config(self) -> C:
        ...

    def save(self, path: PathLike[str], compile_model: bool = False) -> None:
        ...

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        ...

    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor]) -> None:
        ...
