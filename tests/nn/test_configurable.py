# pylint: disable=missing-class-docstring,missing-function-docstring
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from torch import nn
from lightkit.nn import Configurable


@dataclass
class MyConfig:
    input_dim: int
    output_dim: int
    hidden_layers: List[int] = field(default_factory=list)
    dropout: float = 0.1


class MyMLP(Configurable[MyConfig], nn.Sequential):
    def __init__(self, config: MyConfig):
        layers = []
        dims = zip(
            [config.input_dim] + config.hidden_layers,
            config.hidden_layers + [config.output_dim],
        )
        for i, (in_dim, out_dim) in enumerate(dims):
            if i > 0:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout))
            layers.append(nn.Linear(in_dim, out_dim))

        super().__init__(config, *layers)


def test_save_load():
    config = MyConfig(10, 1, hidden_layers=[8, 8])
    mlp = MyMLP(config)

    with tempfile.TemporaryDirectory() as tmp:
        mlp.save(tmp)

        assert len(os.listdir(tmp)) == 2
        assert (Path(tmp) / "parameters.pt").exists()
        assert (Path(tmp) / "config.json").exists()

        with (Path(tmp) / "config.json").open("r") as f:
            loaded_config = json.load(f)
            assert config == MyConfig(**loaded_config)

        loaded_mlp = MyMLP.load(tmp)
        assert loaded_mlp.config == mlp.config
        # assert loaded_mlp.state_dict() == mlp.state_dict()
