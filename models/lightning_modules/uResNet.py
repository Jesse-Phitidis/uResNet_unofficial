import pytorch_lightning as pl
import torchio as tio
import torch
import torch.nn as nn
import numpy as np
from monai.inferers import SliceInferer 
import nibabel as nib
from os.path import join
from monai.metrics import DiceMetric
from utils.tensor_handling import one_warm_to_hot
from pytorch_lightning.callbacks import ModelCheckpoint

class uResNet_LightningModule(pl.LightningModule):
    def __init__(self, network: nn.Module, criterion, patch_size, write_path):
        super().__init__()

        self.network = network
        self.criterion = criterion
        self.patch_size = patch_size
        self.write_path = write_path

        self.inferer = SliceInferer(roi_size=self.patch_size[:-1], spatial_dim=2)
        self.DiceMetric = DiceMetric(include_background=False, reduction='mean_batch', get_not_nans=True)

        self.save_hyperparameters(ignore=['network', 'criterion'])

    def forward(self, x):
        return self.network(x)

    def extract_data(self, batch):
        return batch['image'][tio.DATA], batch['label'][tio.DATA], batch['id'], batch['image']['affine']

    def training_step(self, batch, batch_idx):
        image, label, id, _ = self.extract_data(batch)
        pred = self.network(image.squeeze(4))
        loss = self.criterion(pred, label.squeeze(4).squeeze(1).type(torch.int64))
        self.log('loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.print('\non_validation_epoch_end was run\n')
        if self.trainer.datamodule.val_run is False:
            return
        image, label, _, _ = self.extract_data(batch)
        pred = self.inferer(image, self.network)
        pred_one_hot = one_warm_to_hot(pred)
        self.DiceMetric(pred_one_hot, label)

    def on_validation_epoch_end(self):
        if self.trainer.datamodule.val_run is False:
            return
        (wmh_dice, stroke_dice), (wmh_not_nan, stroke_not_nan) = self.DiceMetric.aggregate()
        self.log('wmh_dice', wmh_dice.item(), on_step=False, on_epoch=True)
        self.log('stroke_dice', stroke_dice.item(), on_step=False, on_epoch=True)
        self.log('both_non_nans', {'wmh_not_nan': wmh_not_nan.item(), 'stroke_not_nan': stroke_not_nan.item()}, on_step=False, on_epoch=True)
        self.log('both_dices', {'wmh_dice': wmh_dice.item(), 'stroke_dice': stroke_dice.item()}, on_step=False, on_epoch=True)
        self.DiceMetric.reset()

    def predict_step(self, batch, batch_idx):
        image, _, id, affine = self.extract_data(batch)
        pred_soft = self.inferer(image, self.network)
        pred_hard3D = torch.squeeze(torch.argmax(pred_soft, dim=1))
        pred_nifti_image = nib.Nifti1Image(pred_hard3D.cpu().numpy().astype(np.uint8), affine=affine[0].cpu().numpy().astype(np.float64))
        name = 'MSS_' + id[0] + '.nii.gz'
        save_path = join(self.write_path, name)
        nib.save(pred_nifti_image, save_path)

    def on_fit_start(self):
        if self.trainer.datamodule.val_run is False:
            self.trainer.limit_val_batches = 0
            self.trainer.num_sanity_val_steps = 0



