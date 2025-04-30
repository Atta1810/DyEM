# To Do List

# [] unbatched predictions for calculating metrics

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger


def get_callbacks(configs):
    callbacks = {}
    if "model_checkpoint" in configs:
        callbacks["model_checkpoint"] = ModelCheckpoint(
            monitor=configs.model_checkpoint.monitor,  # Direct access
            mode=configs.model_checkpoint.mode,
            save_top_k=configs.model_checkpoint.save_top_k
        )
    if "early_stopping" in configs:
        callbacks["early_stopping"] = EarlyStopping(
            monitor=configs.early_stopping.monitor,  # Direct access
            patience=configs.early_stopping.patience,
            mode=configs.early_stopping.mode
        )
    return callbacks

def get_logger(configs):
    return CSVLogger("logs", name=configs)
