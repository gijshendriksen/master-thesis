import re
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer

from information_extraction.dtypes import BertBatch
from information_extraction.training.data import BaseDataset
from information_extraction.data.metrics import normalize_answer, normalize_with_mapping


class BertDataset(BaseDataset):
    start_char_positions: List[int]
    end_char_positions: List[int]
    encoded_ancestors: Dict[str, torch.Tensor]
    empty_encoding: torch.Tensor

    def prepare_inputs(self):
        self.start_char_positions = []
        self.end_char_positions = []

        not_null_indices = []
        num_not_found = 0
        for index, (context, target) in enumerate(zip(self.inputs, self.targets)):
            normalized_target = normalize_answer(target)

            if not normalized_target:
                # Use -1 to indicate the value was not found
                self.start_char_positions.append(-1)
                self.end_char_positions.append(-1)
                continue

            # We find the normalized answer in the normalized context, and then map that back to the original sequence
            normalized_context, char_mapping = normalize_with_mapping(' '.join(context))

            match = re.search(f'\\b{re.escape(normalized_target)}\\b', normalized_context)

            if match is not None and 0 <= match.start() <= match.end() - 1 < len(char_mapping):
                self.start_char_positions.append(char_mapping[match.start()])
                self.end_char_positions.append(char_mapping[match.end() - 1])
                not_null_indices.append(index)
            else:
                # Use -1 to indicate the value was not found
                self.start_char_positions.append(-1)
                self.end_char_positions.append(-1)
                num_not_found += 1

        if self.remove_null:
            self.inputs = [self.inputs[i] for i in not_null_indices]
            self.targets = [self.targets[i] for i in not_null_indices]
            self.ancestors = [self.ancestors[i] for i in not_null_indices]
            self.features = [self.features[i] for i in not_null_indices]
            self.start_char_positions = [self.start_char_positions[i] for i in not_null_indices]
            self.end_char_positions = [self.end_char_positions[i] for i in not_null_indices]
        elif num_not_found > 0:
            print(f'Warning: BertDataset found {num_not_found}/{len(self.inputs)} samples '
                  f'where the context does not contain the answer!')

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        distinct_ancestors = list(set(x for a in self.ancestors for x in a))
        encoded_ancestors = model.encode(distinct_ancestors, device=device, convert_to_numpy=True,
                                         show_progress_bar=True)

        self.encoded_ancestors = {
            ancestor: torch.Tensor(encoding)
            for ancestor, encoding in zip(distinct_ancestors, encoded_ancestors)
        }
        self.empty_encoding = torch.zeros(encoded_ancestors[0].shape)

    def __getitem__(self, idx: List[int]) -> BertBatch:
        docs = [self.docs[i] for i in idx]
        inputs = [self.inputs[i] for i in idx]
        ancestors = [self.ancestors[i] for i in idx]
        targets = [self.targets[i] for i in idx]
        features = [self.features[i] for i in idx]
        start_char_positions = [self.start_char_positions[i] for i in idx]
        end_char_positions = [self.end_char_positions[i] for i in idx]

        encoding = self.tokenizer([[f] for f in features], inputs, **self.tokenize_kwargs)

        start_positions = []
        end_positions = []

        for i, (start_char, end_char) in enumerate(zip(start_char_positions, end_char_positions)):
            if start_char < 0:
                # In this case, the answer does not exist in the context
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_pos = encoding.char_to_token(i, start_char, sequence_index=1)
                end_pos = encoding.char_to_token(i, end_char, sequence_index=1)

                start_word = encoding.char_to_word(i, start_char, 1)
                end_word = encoding.char_to_word(i, end_char, 1)

                if start_pos is not None and end_pos is not None:
                    start_positions.append(start_pos)
                    end_positions.append(end_pos)

                    sentence = self.tokenizer.decode(encoding.input_ids[i, start_pos:end_pos + 1])

                    if 'offered' in sentence:
                        print(inputs[i])
                        print('CHECK SPAN:', targets[i], '-', start_pos, end_pos,
                              sentence, '-',
                              ' '.join(inputs[i])[start_char:end_char+1], '-',
                              inputs[i][start_word:end_word + 1])
                        break

                else:
                    start_positions.append(0)
                    end_positions.append(0)

        encoded_ancestors = []
        for i, ancestor in enumerate(ancestors):
            current_ancestors = []
            for j, token in enumerate(encoding.input_ids[i]):
                word_index = encoding.token_to_word(i, j)
                if encoding.token_to_sequence(i, j) == 1 and word_index is not None:
                    # print(self.tokenizer.convert_ids_to_tokens([token])[0], word_index)
                    current_ancestors.append(self.encoded_ancestors[ancestor[word_index]])
                else:
                    current_ancestors.append(self.empty_encoding)

            encoded_ancestors.append(torch.stack(current_ancestors))

        encoded_ancestors = pad_sequence(encoded_ancestors, batch_first=True)

        return BertBatch(
            docs,
            inputs,
            encoded_ancestors,
            targets,
            features,
            encoding.input_ids,
            encoding.attention_mask,
            encoding.token_type_ids,
            torch.as_tensor(start_positions),
            torch.as_tensor(end_positions),
        )
