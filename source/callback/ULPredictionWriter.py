from pathlib import Path
from typing import Any, List, Sequence, Optional

import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor


class ULPredictionWriter(BasePredictionWriter):


    def __init__(self, params):
        super(ULPredictionWriter, self).__init__(params.write_interval)
        self.params=params

    def write_on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", predictions: Sequence[Any],
                           batch_indices: Optional[Sequence[Any]]) -> None:
        pass


    def write_on_batch_end(
            self, trainer, pl_module, prediction: Any, batch_indices: List[int], batch: Any,
            batch_idx: int, dataloader_idx: int
    ):

        predictions = []

        for idx, desc_repr, code_repr in zip(
                prediction["idx"].tolist(),
                prediction["desc_repr"].tolist(),
                prediction["code_repr"].tolist()
        ):


            predictions.append({
                "idx": idx,
                "desc_repr": desc_repr,
                "code_repr": code_repr
            })

        self._checkpoint(predictions, dataloader_idx, batch_idx)



    def _checkpoint(self, predictions, dataloader_idx, batch_idx):
        dir = f"{self.params.dir}fold_{self.params.fold}/"
        Path(dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            predictions,
            f"{dir}{dataloader_idx}_{batch_idx}.prd"
        )

