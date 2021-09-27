import json
import pickle

import torch
from torch.utils.data import Dataset


class PawsDataset(Dataset):
    """Paws Dataset.
    """

    def __init__(self, samples, ids_path, st1_tokenizer, st2_tokenizer, st1_max_length, st2_max_length):
        super(PawsDataset, self).__init__()
        self.samples = samples
        self.st1_tokenizer = st1_tokenizer
        self.st2_tokenizer= st2_tokenizer
        self.st1_max_length= st1_max_length
        self.st2_max_length=st2_max_length
        self._load_ids(ids_path)

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            self.ids = pickle.load(ids_file)

    def _encode(self, sample):
        return {
            "idx": sample["idx"],
            "st1": torch.tensor(
                self.st1_tokenizer.encode(text=sample["st1"], max_length=self.st1_max_length, padding="max_length",
                                          truncation=True)
            ),
            "st2": torch.tensor(
                self.st2_tokenizer.encode(text=sample["st2"], max_length=self.st2_max_length, padding="max_length",
                                          truncation=True)
            ),
            "cls": sample["cls"],
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        return self._encode(
            self.samples[sample_id]
        )
