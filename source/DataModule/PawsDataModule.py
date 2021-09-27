import pickle

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.Dataset.PawsDataset import PawsDataset


class PawsDataModule(pl.LightningDataModule):
    """
    Paws DataModule
    """

    def __init__(self, params, st1_tokenizer, st2_tokenizer, fold):
        super(PawsDataModule, self).__init__()
        self.params = params
        self.st1_tokenizer = st1_tokenizer
        self.st2_tokenizer = st2_tokenizer
        self.fold = fold

    def prepare_data(self):
        with open(self.params.dir + f"samples.pkl", "rb") as dataset_file:
            self.samples = pickle.load(dataset_file)

    def setup(self, stage=None):

        if stage == 'fit':
            self.train_dataset = PawsDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold}/train.pkl",
                st1_tokenizer=self.st1_tokenizer,
                st2_tokenizer=self.st2_tokenizer,
                st1_max_length=self.params.st1_max_length,
                st2_max_length=self.params.st2_max_length
            )

            self.val_dataset = PawsDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold}/val.pkl",
                st1_tokenizer=self.st1_tokenizer,
                st2_tokenizer=self.st2_tokenizer,
                st1_max_length=self.params.st1_max_length,
                st2_max_length=self.params.st2_max_length
            )

        if stage == 'test' or stage is "predict":
            self.test_dataset = PawsDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold}/val.pkl",
                st1_tokenizer=self.st1_tokenizer,
                st2_tokenizer=self.st2_tokenizer,
                st1_max_length=self.params.st1_max_length,
                st2_max_length=self.params.st2_max_length
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers
        )

    def predict_dataloader(self):
        return self.test_dataloader()
