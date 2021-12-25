from pathlib import Path
from typing import Generic, TypeVar
from lightkit.nn._protocols import ConfigurableModule
from lightkit.utils import get_generic_type
from .base import BaseEstimator
from .exception import NotFittedError

M = TypeVar("M", bound=ConfigurableModule)  # type: ignore


class ConfigurableBaseEstimator(BaseEstimator, Generic[M]):
    """
    Extension of the base estimator which allows to manage a single model that uses the
    :class:`lightkit.nn.Configurable` mixin.
    """

    @property
    def model_(self) -> M:
        """
        The fitted PyTorch module containing all estimated parameters.
        """
        if not hasattr(self, "_model"):
            raise NotFittedError(f"`{self.__class__.__name__}` has not been fitted yet")
        return self._model

    @model_.setter
    def model_(self, model: M) -> None:
        self._model = model

    def save_attributes(self, path: Path) -> None:
        # First, store simple attributes
        super().save_attributes(path)

        # Then, store the model
        self.model_.save(path / "model")

    def load_attributes(self, path: Path) -> None:
        # First, load simple attributes
        super().load_attributes(path)

        # Then, load the model
        model_cls = get_generic_type(self.__class__, ConfigurableBaseEstimator)
        self.model_ = model_cls.load(path / "model")  # type: ignore
