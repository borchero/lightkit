from __future__ import annotations
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Generic, Type
import torch
from torch import jit
from lightkit.utils import get_generic_type
from ._protocols import C, ConfigurableModule, M, PathType


class Configurable(Generic[C]):
    """
    A mixin for any PyTorch module to extend it with storage capabilities. By passing a single
    configuration object to the initializer, this mixin allows the module to be extended with
    :meth:`save` and :meth:`load` methods. These methods allow to (1) save the model along with
    its configuration (i.e. architecture) and (2) to load the model without instantiating an
    instance of the class.
    """

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

        config_cls = get_generic_type(cls, Configurable)
        with (path / "config.json").open("r") as f:
            config_args = json.load(f)
            config = _init_config(config_cls, config_args)

        model = cls(config)  # type: ignore
        with (path / "parameters.pt").open("rb") as f:
            state_dict = torch.load(f)
        model.load_state_dict(state_dict)  # pylint: disable=no-member
        return model


def _init_config(target: Type[Any], args: Dict[str, Any]) -> Any:
    result = {}
    for key, val in args.items():
        arg_type = target.__dataclass_fields__[key].type  # type: ignore
        if dataclasses.is_dataclass(arg_type):
            result[key] = _init_config(arg_type, val)
        else:
            result[key] = val
    return target(**result)
