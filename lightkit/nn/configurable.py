from __future__ import annotations
import dataclasses
import json
from pathlib import Path
from typing import Any, Generic, get_args, get_origin, Type
import torch
from torch import jit
from ._protocols import C, ConfigurableModule, M, PathType


class Configurable(Generic[C]):
    """
    A mixin for any PyTorch module to extend it with storage capabilities. By passing a single
    configuration object to the initializer, this mixin allows the module to be extended with
    :meth:`save` and :meth:`load` methods. These methods allow to to (1) save the model along with
    its configuration (i.e. architecture) and (2) to load the model without instantiating an
    instance of the class.
    """

    @classmethod
    def load(cls: Type[M], path: PathType) -> M:
        """
        Loads the module's configurations and parameters from files in the specified directory at
        first. Then, it initializes the model with the stored configurations and loads the
        parameters. This method is typically used after calling :meth:`save` on the model.

        Args:
            path: The directory which contains the ``config.json`` and ``parameters.pt`` files to
                load.

        Returns:
            The loaded model.

        Note:
            You can load modules even after you changed their configuration class. The only
            requirement is that any new configuration options have a default value.
        """
        path = Path(path)
        assert path.is_dir(), "Modules can only be loaded from a directory."

        config_cls = cls._get_config_class()  # type: ignore
        with (path / "config.json").open("r") as f:
            config = config_cls(**json.load(f))

        model = cls(config)
        with (path / "parameters.pt").open("rb") as f:
            state_dict = torch.load(f)
        model.load_state_dict(state_dict)  # pylint: disable=no-member
        return model

    @classmethod
    def _get_config_class(cls: Type[M]) -> Type[C]:
        # pylint: disable=no-member
        for base in cls.__orig_bases__:  # type: ignore
            if get_origin(base) == Configurable:
                args = get_args(base)
                if not args:
                    raise ValueError(
                        f"`{cls.__name__} does not provide a generic parameter for `Configurable`"
                    )
                return get_args(base)[0]
        raise ValueError(f"`{cls.__name__}` does not inherit from `Configurable`")

    def __init__(self, config: C, *args: Any, **kwargs: Any):
        """
        Args:
            config: The configuration of the architecture.
            args: Positional arguments that ought to be passed to the superclass.
            kwargs: Keyword arguments that ought to be passed to the superclass.
        """
        assert dataclasses.is_dataclass(config), "Configuration is not a dataclass."

        super().__init__(*args, **kwargs)
        self.config = config

    @jit.unused
    def save(self: ConfigurableModule[C], path: PathType, compile_model: bool = False) -> None:
        """
        Saves the module's configuration and parameters to files in the specified directory. It
        creates two files, namely ``config.json`` and ``parameters.pt`` which contain the
        configuration and parameters, respectively.

        Args:
            path: The directory to which to save the configuration and parameter files. The
                directory may or may not exist but no parent directories are created.
            compile_model: Whether the model should be compiled via TorchScript. An additional file
                called ``model.ptc`` will then be stored. Note that you can simply load the
                compiled model via :meth:`torch.jit.load` at a later point.
        """
        path = Path(path)
        assert path.is_dir(), "Modules can only be saved to a directory."

        path.mkdir(parents=False, exist_ok=True)
        with (path / "config.json").open("w+") as f:
            json.dump(dataclasses.asdict(self.config), f)
        with (path / "parameters.pt").open("wb+") as f:
            torch.save(self.state_dict(), f)  # pylint: disable=no-member

        if compile_model:
            compiled_model = jit.script(self)
            with (path / "model.ptc").open("wb+") as f:
                jit.save(compiled_model, f)
