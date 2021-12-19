import pytorch_lightning as pl
from pytorch_lightning.plugins import DataParallelPlugin, DDP2Plugin, DDPSpawnPlugin


def trainer_uses_batch_training(trainer: pl.Trainer) -> bool:
    """
    Returns whether the given trainer's configuration implies batch training. When this function
    returns ``True``, training is certainly performed on batches. If it returns ``False``, it
    depends on the data that is used for training and its batch size.

    Args:
        trainer: The trainer for which to determine whether it possibly uses batch training.

    Returns:
        A boolean indicating whether the trainer possibly uses batch training.
    """
    assert not isinstance(
        trainer.training_type_plugin,
        (DDP2Plugin, DataParallelPlugin, DDPSpawnPlugin),
    ), "Unable to determine whether trainer uses batch training due to unsupported plugin."

    return (
        trainer.num_gpus > 1
        or trainer.num_nodes > 1
        or trainer.num_processes > 1
        or (trainer.tpu_cores is not None and trainer.tpu_cores > 1)
    )
