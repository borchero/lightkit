from __future__ import annotations
import inspect
import logging
import pickle
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Generic, get_args, get_origin, Optional, Type, TypeVar
import pytorch_lightning as pl
from lightkit.data.types import DataLoader
from lightkit.nn._protocols import ConfigurableModule
from ._trainer import LightkitTrainer
from .exception import NotFittedError

M = TypeVar("M", bound=ConfigurableModule)  # type: ignore
E = TypeVar("E", bound="BaseEstimator")  # type: ignore
T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseEstimator(Generic[M], ABC):
    """
    Base estimator class that all estimators should inherit from. This base estimator does not
    enforce the implementation of any methods, but users should follow the Scikit-learn guide on
    implementing estimators (which can be found
    `here <https://scikit-learn.org/stable/developers/develop.html>`_). Some of the methods
    mentioned in this guide are already implemented in this base estimator and work as expected if
    the aspects listed below are followed.

    In contrast to Scikit-learn's estimator, this estimator is strongly typed and integrates well
    with PyTorch Lightning. Most importantly, it provides the :meth:`trainer` method which returns
    a fully configured trainer to be used by other methods. The configuration is stored in the
    estimator and can be adjusted by passing parameters to ``default_params``, ``user_params`` and
    ``overwrite_params`` when calling ``super().__init__()``. By default, the base estimator sets
    the following flags:

    - Logging is disabled (``logger=False``).
    - Logging is performed at every step (``log_every_n_steps=1``).
    - The progress bar is only enabled (``enable_progress_bar``) if LightKit's logging level is
        ``INFO`` or more verbose.
    - Checkpointing is only enabled (``enable_checkpointing``) if LightKit's logging level is
        ``DEBUG`` or more verbose.
    - The model summary is only enabled (``enable_model_summary``) if LightKit's logging level is
        ``DEBUG`` or more verbose.

    Note that the logging level can be changed via :meth:`lightkit.set_logging_level`.

    When subclassing this base estimator, users should take care of the following aspects:

    - All parameters passed to the initializer must be assigned to attributes with the same name.
      This ensures that :meth:`get_params` and :meth:`set_params` work as expected. Parameters that
      are passed to the trainer *must* be named ``trainer_params`` and should not be manually
      assigned to an attribute (this is handled by the base estimator).
    - Fitted attributes must have a single trailing underscore (e.g. ``model_``). They should be
      defined as annotations and carry a description. The trailing underscore ensures that access
      to these attributes results in a useful error message.
    """

    # This is both a private and public property to properly generate documentation.
    _model: M

    def __init__(
        self,
        *,
        default_params: Optional[Dict[str, Any]] = None,
        user_params: Optional[Dict[str, Any]] = None,
        overwrite_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            default_params: Estimator-specific parameters that provide defaults for configuring the
                PyTorch Lightning trainer. An example might be setting ``max_epochs``. Overwrites
                the default parameters established by the base estimator.
            user_params: User-specific parameters that configure the PyTorch Lightning trainer.
                This dictionary should be passed through from a ``trainer_params`` init argument in
                subclasses. Overwrites any of the default parameters.
            overwrite_params: PyTorch Lightning trainer flags that need to be ensured independently
                of user-provided parameters. For example, ``max_epochs`` could be fixed to a
                certain value.
        """
        self.trainer_params_user = user_params
        self.trainer_params = {
            **dict(
                logger=False,
                log_every_n_steps=1,
                enable_progress_bar=logger.getEffectiveLevel() <= logging.INFO,
                enable_checkpointing=logger.getEffectiveLevel() <= logging.DEBUG,
                enable_model_summary=logger.getEffectiveLevel() <= logging.DEBUG,
            ),
            **(default_params or {}),
            **(user_params or {}),
            **(overwrite_params or {}),
        }

    @property
    def model_(self) -> M:
        """
        The fitted PyTorch module containing all estimated parameters.
        """
        return self._model

    def trainer(self, **kwargs: Any) -> pl.Trainer:
        """
        Returns the trainer as configured by the estimator. Typically, this method is only called
        by functions in the estimator.

        Args:
            kwargs: Additional arguments that override the trainer arguments registered in the
                initializer of the estimator.

        Returns:
            A fully initialized PyTorch Lightning trainer.

        Note:
            This function should be preferred over initializing the trainer directly. It ensures
            that the returned trainer correctly deals with LightKit components such as the
            :class:`~lightkit.data.TensorDataLoader`.
        """
        return LightkitTrainer(**{**self.trainer_params, **kwargs})

    # ---------------------------------------------------------------------------------------------
    # PERSISTENCE

    def load_model(self, model: M) -> None:
        """
        Loads the provided model that has been fitted previously by this estimator or manually
        without the use of the estimator.

        Args:
            model: The model to load. In case, this estimator is already fitted, this model
                overrides the existing fitted model.
        """
        self._model = model

    def save(self, path: Path) -> None:
        """
        Saves the estimator to the provided directory. It saves a file named ``estimator.pickle``
        for the configuration of the estimator and additional files for the fitted model (if
        applicable). For more information on the files saved for the fitted model or for more
        customization, look at :meth:`get_params` and :meth:`lightkit.nn.Configurable.save`.

        Args:
            path: The directory to which all files should be saved.

        Note:
            This method may be called regardless of whether the estimator has already been fitted.

        Attention:
            Use this method with care. It uses :mod:`pickle` to store the configuration options of
            the estimator and is thus not necessarily backwards-compatible. Instead, consider
            using :meth:`lighkit.nn.Configurable.save` on the fitted model accessible via
            :attr:`model_`.
        """
        assert path.is_dir(), "Estimators can only be saved to a directory."

        with (path / "estimator.pickle").open("wb+") as f:
            pickle.dump(self.get_params(), f)

        if self._is_fitted:
            self.model_.save(path)

    @classmethod
    def load(cls: Type[E], path: Path) -> E:
        """
        Loads the estimator and (if available) the fitted model. See :meth:`save` for more
        information about the required filenames for loading.

        Args:
            path: The directory from which to load the estimator.

        Returns:
            The loaded estimator, either fitted or not, depending on the availability of the
            ``config.json`` file.
        """
        estimator = cls()
        with (path / "estimator.pickle").open("rb") as f:
            estimator.set_params(pickle.load(f))  # type: ignore

        if (path / "config.json").exists():
            model_cls = cls._get_model_class()
            model = model_cls.load(path)
            estimator.load_model(model)

        return estimator

    # ---------------------------------------------------------------------------------------------
    # SKLEARN INTERFACE

    def get_params(self, deep: bool = True) -> Dict[str, Any]:  # pylint: disable=unused-argument
        """
        Returns the estimator's parameters as passed to the initializer.

        Args:
            deep: Ignored. For Scikit-learn compatibility.

        Returns:
            The mapping from init parameters to values.
        """
        signature = inspect.signature(self.__class__.__init__)
        parameters = [p.name for p in signature.parameters.values() if p.name != "self"]
        return {p: getattr(self, p) for p in parameters}

    def set_params(self: E, values: Dict[str, Any]) -> E:
        """
        Sets the provided values on the estimator. The estimator is returned as well, but the
        estimator on which this function is called is also modified.

        Args:
            values: The values to set.

        Returns:
            The estimator where the values have been set.
        """
        for key, value in values.items():
            setattr(self, key, value)
        return self

    # ---------------------------------------------------------------------------------------------
    # SPECIAL METHODS

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__:
            return self.__dict__[key]
        if key.endswith("_") and not key.endswith("__"):
            raise NotFittedError(f"`{self.__class__.__name__}` has not been fitted yet")
        raise AttributeError(
            f"Attribute `{key}` does not exist on type `{self.__class__.__name__}`."
        )

    # ---------------------------------------------------------------------------------------------
    # PRIVATE

    def _num_batches_per_epoch(self, loader: DataLoader[T]) -> int:
        """
        Returns the number of batches that are run for the given data loader across all processes
        when using the trainer provided by the :meth:`trainer` method. If ``n`` processes run
        ``k`` batches each, this method returns ``k * n``.
        """
        trainer = self.trainer()
        num_batches = len(loader)  # type: ignore
        kwargs = trainer.distributed_sampler_kwargs
        if kwargs is None:
            return num_batches
        return num_batches * kwargs.get("num_replicas", 1)

    @property
    def _is_fitted(self) -> bool:
        try:
            getattr(self, "model_")
            return True
        except NotFittedError:
            return False

    # ---------------------------------------------------------------------------------------------
    # GENERICS

    @classmethod
    def _get_model_class(cls: Type[E]) -> Type[M]:
        return cls._get_generic_type(0)

    @classmethod
    def _get_generic_type(cls: Type[E], index: int) -> Any:
        for base in cls.__orig_bases__:  # type: ignore
            if get_origin(base) == BaseEstimator:
                args = get_args(base)
                if not args:
                    raise ValueError(
                        f"`{cls.__name__} does not provide at least {index+1} generic parameters"
                        " for `Estimator`"
                    )
                return get_args(base)[index]
        raise ValueError(f"`{cls.__name__}` does not inherit from `Estimator`")
