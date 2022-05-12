from collections import namedtuple
from typing import List, Optional

import numpy as np

from transformers import T5Tokenizer

from data.base import BaseDataset

T5Batch = namedtuple('T5Batch', ['inputs', 'targets', 'features',
                                 'input_ids', 'attention_mask', 'decoder_input_ids',
                                 'decoder_attention_mask', 'target_labels'])


class T5Dataset(BaseDataset):
    input_ids: np.array
    attention_mask: np.array
    decoder_input_ids: np.array
    decoder_attention_mask: np.array
    target_labels: np.array

    def __init__(self, files: List[str], t5_version: str, remove_null: bool = False, max_length: Optional[int] = None):
        self.tokenizer = T5Tokenizer.from_pretrained(t5_version)
        super().__init__(files, remove_null, max_length)

    def prepare_inputs(self):
        formatted_inputs = [
            f'attribute: {feature} context: {text}'
            for text, feature in zip(self.inputs, self.features)
        ]

        inputs = self.tokenizer(formatted_inputs, **self.tokenize_kwargs)
        targets = self.tokenizer(self.targets, **self.tokenize_kwargs)

        self.input_ids = inputs.input_ids
        self.attention_mask = inputs.attention_mask
        self.decoder_input_ids = targets.input_ids[:, :-1]
        self.decoder_attention_mask = targets.attention_mask[:, :-1]
        self.target_labels = targets.input_ids[:, 1:]

    def __getitem__(self, item: int) -> T5Batch:
        return T5Batch(
            self.inputs[item],
            self.targets[item],
            self.features[item],
            self.input_ids[item],
            self.attention_mask[item],
            self.decoder_input_ids[item],
            self.decoder_attention_mask[item],
            self.target_labels[item],
        )