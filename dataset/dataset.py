# 
# base module for LLM task
# GPTDateset: input is max_length sub_sequence and target shift 1 based input sequence
#
import json
import pandas as pd
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self
            ,text
            ,tokenizer
            ,max_length
            ,stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        if len(token_ids) < max_length - 1:
            raise ValueError(f"token size({len(token_ids)}) is less then max_length({max_length}).")

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    @classmethod
    def from_file(cls
            ,file
            ,tokenizer
            ,max_length
            ,stride
            ,encoding="utf-8"):
        with open(file, "r", encoding=encoding) as f:
            raw_text = f.read()

        return cls(raw_text, tokenizer, max_length, stride)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}")

        return self.input_ids[idx], self.target_ids[idx]


class GPTDataLoader():
    def __init__(self, tokenizer, num_workers=0):
        self.tokenizer = tokenizer
        self.num_workers = num_workers
  
    def _create(self
            ,dataset
            ,batch_size
            ,shuffle
            ,drop_last
            ,num_workers):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers
        )

    def text_dataloader(self
            ,text 
            ,batch_size=4
            ,max_length=256
            ,stride=128
            ,shuffle=True
            ,drop_last=True):

        dataset = GPTDataset(text, self.tokenizer, max_length, stride)
        
        return self._create(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers
        )

    def file_dataloader(self
            ,file
            ,encoding="utf-8"
            ,batch_size=4
            ,max_length=256
            ,stride=128
            ,shuffle=True
            ,drop_last=True):
        
        dataset = GPTDataset.from_file(
            file                        
            ,self.tokenizer
            ,max_length
            ,stride
            ,encoding
        )
        
        return self._create(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers
        )

    def text_train_val_dataloader(self
            ,text
            ,train_ratio=0.9
            ,batch_size=4
            ,max_length=256
            ,stride=128):
        assert train_ratio > 0.0 and train_ratio < 1.0, "train_ratio must be large than 0.0 and less then 1.0"
        split_idx = int(len(text) * train_ratio)
        train_text = text[:split_idx]
        val_text = text[split_idx:]

        train_dataset = GPTDataset(train_text, self.tokenizer, max_length, stride)
        val_dataset = GPTDataset(val_text, self.tokenizer, max_length, stride)
        
        train_loader = self._create(
            train_dataset
            ,batch_size=batch_size
            ,shuffle=True
            ,drop_last=True
            ,num_workers=self.num_workers
        ) 

        val_loader = self._create(
            val_dataset
            ,batch_size=batch_size
            ,shuffle=False
            ,drop_last=False
            ,num_workers=self.num_workers
        )

        return train_loader, val_loader

    def file_train_val_dataloader(self
            ,file
            ,encoding="utf-8"
            ,train_ratio=0.9
            ,batch_size=4
            ,max_length=256
            ,stride=128):
        with open(file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        return self.text_train_val_dataloader(
            raw_text
            ,train_ratio
            ,batch_size
            ,max_length
            ,stride
        )


class LabeledDataset(Dataset):
    def __init__(self
            ,csv_file
            ,tokenizer
            ,max_length=None
            ,pad_token_id=None
            ,text_field="Text"):
        self.data = pd.read_csv(csv_file)
        self.encoded_ids = [
            tokenizer.encode(text) for text in self.data[text_field]
        ]

        if pad_token_id is None:
            pad_token_id = tokenizer.eos_id()

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_ids = [
                encoded_id[:self.max_length] for encoded_id in self.encoded_ids
            ]

        self.encoded_ids = [
            encoded_id + [pad_token_id] * (self.max_length - len(encoded_id))
            for encoded_id in self.encoded_ids
        ]

    def __getitem__(self, index, label_field="Label"):
        encoded_id = self.encoded_ids[index]
        label = self.data.iloc[index][label_field]
        
        return (
            torch.tensor(encoded_id, dtype=torch.long)
            , torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        return max([len(encoded_id) for encoded_id in self.encoded_ids])

class InstructionDataset(Dataset):
    def __init__(self, json_file, tokenizer, file_encoding="utf-8"):
        with open(json_file, "r", encoding=file_encoding) as f:
            self.data = json.load(f)
        self.encoded_ids = []

        for entry in self.data:
            full_text = self.format_input(entry, with_output=True)
            self.encoded_ids.append(
                tokenizer.encode(full_text)
            )

    @staticmethod
    def format_input(entry, with_output=False):
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )       
        input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
        response_text = f"\n\n### Response:\n{entry['output']}" if with_output and entry["output"] else ""
        return instruction_text + input_text + response_text

    def __getitem__(self, index):
        return self.encoded_ids[index]

    def __len__(self):
        return len(self.data)


def _INSTRUCTION_COLLATE_FN(
            batch,
            pad_token_id=50256,
            ignore_index=-100,
            allowed_max_length=None,
            device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id] #add <|endoftext|> token
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )

        inputs = torch.tensor(padded[:-1]) #truncate the last token
        targets = torch.tensor(padded[1:]) #shift +1 to the right for targets

        mask = (targets == pad_token_id)
        indices = torch.nonzero(mask).squeeze()

        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

INSTRUCTION_COLLATE_FN = partial(
    _INSTRUCTION_COLLATE_FN,
    allowed_max_length=1024
)
