import importlib

import torch
from pytorch_lightning.core.lightning import LightningModule
from hydra.utils import instantiate

from source.metric.SLMRRMetric import SLMRRMetric


class SLESOModel(LightningModule):
    """Supervised Learning Every Step Optimization Model."""

    def __init__(self, hparams):

        super(SLESOModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoders
        self.desc_encoder = instantiate(hparams.desc_encoder)
        self.code_encoder = instantiate(hparams.code_encoder)

        # loss function
        self.loss = instantiate(hparams.loss)

        # metric
        self.mrr = SLMRRMetric()


    def forward(self, desc, code):
        desc_repr = self.desc_encoder(desc)
        code_repr = self.code_encoder(code)
        return desc_repr, code_repr

    def training_step(self, batch, batch_idx):
        desc, code, cls = batch["desc"], batch["code"], batch["cls"]
        desc_repr, code_repr = self(desc, code)
        train_loss=self.loss(desc_repr, code_repr, cls)
        self.log("train_LOSS", train_loss, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        desc, code, cls = batch["desc"], batch["code"], batch["cls"]
        desc_repr, code_repr = self(desc, code)
        self.log("val_MRR", self.mrr(desc_repr, code_repr, cls), prog_bar=True)
        self.log("val_LOSS", self.loss(desc_repr, code_repr, cls), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.mrr.compute()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        idx, desc, code, cls = batch["idx"], batch["desc"], batch["code"], batch["cls"]
        desc_repr, code_repr = self(desc, code)

        return {
            "idx": idx,
            "desc_repr": desc_repr,
            "code_repr": code_repr,
            "cls": cls
        }

    def test_step(self, batch, batch_idx):
        desc, code, cls = batch["desc"], batch["code"], batch["cls"]
        desc_repr, code_repr = self(desc, code)
        self.log("test_MRR", self.mrr(desc_repr, code_repr, cls), prog_bar=True)

    def test_epoch_end(self, outs):
        self.mrr.compute()

    def get_desc_encoder(self):
        return self.desc_encoder

    def get_code_encoder(self):
        return self.desc_encoder


    def configure_optimizers(self):
        # optimizers
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-08,
                              weight_decay=self.hparams.weight_decay, amsgrad=True)

        # schedulers
        step_size_up = round(0.03 * self.num_training_steps)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode='triangular2', base_lr=self.hparams.base_lr,
                                              max_lr=self.hparams.max_lr, step_size_up=step_size_up,
                                              cycle_momentum=False)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and number of epochs."""
        steps_per_epochs = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        max_epochs = self.trainer.max_epochs
        return steps_per_epochs * max_epochs
