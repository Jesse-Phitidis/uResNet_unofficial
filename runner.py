import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
import os
import sys

MODEL_DEVELOPMENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(MODEL_DEVELOPMENT_DIR_PATH)

if __name__ == '__main__':
    cli = LightningCLI(
        pl.LightningModule, 
        pl.LightningDataModule, 
        subclass_mode_model=True, 
        subclass_mode_data=True,
        parser_kwargs={"parser_mode": "omegaconf"}
        )