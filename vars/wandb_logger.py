from pytorch_lightning.loggers import WandbLogger
import wandb


def logger():
    wandb.login(key="")  # insert your key
    logger = WandbLogger(log_model="all")
    return logger
