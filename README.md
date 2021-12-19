# LightKit

LightKit is the missing link between PyTorch and PyTorch Lightning. It allows for easily exposing
your PyTorch model with a fully-fledged Scikit-learn interface.

## Features

**Data:**

- Fast data loader for tabular datasets

## Modules

````python
from dataclasses import dataclass

@dataclass
class MLPConfig:

    num_layers: int
    num_hidden_units: int = 30


```python
from lightkit.nn import Configurable
from torch import nn

class MLP(Configurable[nn.Module):

    def __init__(self, config: MLPConfig):
        super().__init__(self.config)


````
