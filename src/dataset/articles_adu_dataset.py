from typing import List, Optional, Tuple

import json
import warnings
from torch.utils.data import Dataset
import numpy as np

from utils.io import read_excel


class ArticlesADUDataset(Dataset):
    labels: List[str] = ["Fact", "Policy", "Value(-)", "Value", "Value(+)"]

    def __init__(self, adu_excel: str, slice : Optional[List[int]] = None, article_excel: Optional[str] = None, data_directory: str = "./", use_transform: bool = True, transform_p: float = 0.5, use_synonyms: bool = True, synonyms_json: Optional[str] = None) -> None:
        # Files
        self.data_dir = data_directory

        self.adu_file = adu_excel

        self.use_articles = article_excel is not None
        self.article_file = article_excel

        self.synonnyms_file = synonyms_json

        # Transformations
        self.transform_p = transform_p
        self.use_transform = use_transform
        self.use_synonyms = use_synonyms

        # Load data
        self.adus = read_excel(self.adu_file, directory=self.data_dir)
        if slice is not None:
            self.adus = self.adus.iloc[slice]
        self.adus["label"] = self.adus["label"].apply(
            lambda label: self.labels.index(label))

        if self.use_articles:
            self.articles = read_excel(
                self.article_file, directory=self.data_dir)

        if self.use_transform and self.use_synonyms:
            if self.synonnyms_file is None:
                warnings.warn(
                    "use_transform is enabled and use_synonyms is enabled, however synonyms_json isn't defined. Skipping synonyms transformation")
                self.use_synonyms = False
            else:
                with open(f"{self.data_dir}/{self.synonnyms_file}", "r", encoding="utf-8") as synonyms_json:
                    self.synonyms = json.load(synonyms_json)

    def __len__(self) -> int:
        return len(self.adus)

    def __add_synonyms(self, adu: str, p: float = 0.05) -> str:
        words = adu.split(" ")
        for i in range(len(words)):
            word = words[i]
            if (word in self.synonyms) and np.random.binomial(1, p):
                words[i] = np.random.choice(self.synonyms[word])
        return " ".join(words)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        adu = self.adus.iloc[index]

        adu_tokens, label = adu[["tokens", "label"]]

        if self.use_transform and np.random.binomial(1, self.transform_p):
            if self.use_synonyms:
                adu_tokens = self.__add_synonyms(adu_tokens, p=0.05)

        return adu_tokens, label
