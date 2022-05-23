from collections import namedtuple
from typing import List

import torch

from data.base import BaseDataset

BertBatch = namedtuple('BertBatch', ['docs', 'inputs', 'targets', 'features',
                                     'input_ids', 'attention_mask', 'token_type_ids',
                                     'start_positions', 'end_positions'])


class BertDataset(BaseDataset):
    start_char_positions: List[int]
    end_char_positions: List[int]

    def prepare_inputs(self):
        self.start_char_positions = []
        self.end_char_positions = []

        not_null_indices = []
        num_not_found = 0
        for index, (context, target) in enumerate(zip(self.inputs, self.targets)):
            if not target:
                # Use -1 to indicate the value was not found
                self.start_char_positions.append(-1)
                self.end_char_positions.append(-1)
                continue

            try:
                # TODO: is this not too strict?
                start_position = context.index(target)
                end_position = start_position + len(target) - 1

                self.start_char_positions.append(start_position)
                self.end_char_positions.append(end_position)

                not_null_indices.append(index)

            except ValueError:
                # Use -1 to indicate the value was not found
                self.start_char_positions.append(-1)
                self.end_char_positions.append(-1)
                num_not_found += 1

        if self.remove_null:
            self.inputs = [self.inputs[i] for i in not_null_indices]
            self.targets = [self.targets[i] for i in not_null_indices]
            self.features = [self.features[i] for i in not_null_indices]
            self.start_char_positions = [self.start_char_positions[i] for i in not_null_indices]
            self.end_char_positions = [self.end_char_positions[i] for i in not_null_indices]
        elif num_not_found > 0:
            print(f'Warning: BertDataset found {num_not_found}/{len(self.inputs)} samples '
                  f'where the context does not contain the answer!')

    def __getitem__(self, idx: List[int]) -> BertBatch:
        docs = [self.docs[i] for i in idx]
        inputs = [self.inputs[i] for i in idx]
        targets = [self.targets[i] for i in idx]
        features = [self.features[i] for i in idx]
        start_char_positions = [self.start_char_positions[i] for i in idx]
        end_char_positions = [self.end_char_positions[i] for i in idx]

        encoding = self.tokenizer(features, inputs, **self.tokenize_kwargs)

        start_positions = []
        end_positions = []

        for i, (start_char, end_char) in enumerate(zip(start_char_positions, end_char_positions)):
            if start_char < 0:
                # In this case, the answer does not exist in the context
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_positions.append(encoding.char_to_token(i, start_char, sequence_index=1))
                end_positions.append(encoding.char_to_token(i, end_char, sequence_index=1))

        return BertBatch(
            docs,
            inputs,
            targets,
            features,
            encoding.input_ids,
            encoding.attention_mask,
            encoding.token_type_ids,
            torch.as_tensor(start_positions),
            torch.as_tensor(end_positions),
        )
