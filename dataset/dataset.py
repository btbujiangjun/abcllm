# 
# base module for LLM task
# GPTDateset: input is max_length sub_sequence and target shift 1 based input sequence
#
import os
import json
import warnings
from typing import List
import pandas as pd
import numpy as np
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self
            ,token_ids
            ,max_length=256
            ,stride=1):
        if len(token_ids) < max_length - 1:
            raise ValueError(f"token size({len(token_ids)}) is less then max_length({max_length}).")
       
        self.max_length = max_length
        self.stride = stride

        self.token_ids = token_ids
        self.token_size = len(token_ids)
        self.len = int((len(token_ids) - max_length) / stride) + 1

    @classmethod
    def from_text(cls
            ,text:str
            ,tokenizer
            ,max_length=256
            ,stride=1):
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        return cls(token_ids, max_length, stride)

    @classmethod
    def from_files(cls
            ,files:List[str]
            ,tokenizer
            ,max_length=256
            ,stride=1
            ,encoding="utf-8"):
        text = []
        for file in files:
            if os.path.isfile(file):
                with open(file, "r", encoding=encoding) as f:
                    text.append(f.read())
            else:
                warnings.warn(f"{file} is not exists and skip it.")
        
        text = "".join(text)
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        
        return cls(token_ids, max_length, stride)
    
    @classmethod
    def from_preprocess_files(cls
            ,preprocess_files: List[str]
            ,max_length=256
            ,stride=1
            ,memmap=False
            ,dtype="uint16"):
        assert len(preprocess_files) >= 1, f"preprocess file is none."
        
        token_ids = []
        if memmap:
            if len(preprocess_files) > 1:
                warnings.warn(f"memmap mode only support the first file:{preprocess_files[0]}.")
            with open(preprocess_files[0], 'r') as f:
                nbytes = f.seek(0, 2)
                token_size = f.tell() // np.dtype(dtype).itemsize
            token_ids = np.memmap(preprocess_files[0], mode='r', dtype=dtype, shape=(token_size,))
        else:
            for file in preprocess_files:
                if os.path.isfile(file):
                    with open(file,'rb') as f:
                        token_ids.extend(np.fromfile(f, dtype=np.dtype(dtype)))
                else:
                    warnings.warn(f"{file} is not exists and skip it.")
        
        return cls(token_ids, max_length, stride)
                

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if(idx >= len(self)):
            raise IndexError(f"Index {idx} is Out Of Bounds for dataset of size {len(self)}.")
        
        start_index = idx * self.stride
        input_batch = torch.tensor(
            self.token_ids[start_index: start_index + self.max_length]
            , dtype=torch.int32
        )
        target_batch = torch.tensor(
            self.token_ids[start_index + 1 : start_index + 1 + self.max_length]
            , dtype=torch.int32
        )
        
        return input_batch, target_batch


class ABCDataLoader(DataLoader):
    def __init__(self, *args, token_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_size = token_size

class GPTDataLoader():
    def __init__(self, tokenizer, num_workers=0):
        self.tokenizer = tokenizer
        self.num_workers = num_workers

    @property
    def token_size()->int:
        return self.total_token

    def _create(self
            ,dataset
            ,batch_size
            ,shuffle
            ,drop_last
            ,num_workers):
        return ABCDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            token_size=dataset.token_size
        )

    def text_dataloader(self
            ,text: str 
            ,batch_size=4
            ,max_length=256
            ,stride=128
            ,shuffle=True
            ,drop_last=True):
        dataset = GPTDataset.from_text(text, self.tokenizer, max_length, stride)
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
            ,stride=256
            ,shuffle=False
            ,drop_last=True):

        if not isinstance(file, list):
            file = [file]

        dataset = GPTDataset.from_files(
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

    def preprocess_file_dataloader(self
            ,preprocess_files: List[str]
            ,batch_size=4
            ,max_length=256
            ,stride=256
            ,shuffle=False
            ,drop_last=True
            ,memmap=True
            ,dtype="uint16"):
        
        dataset = GPTDataset.from_preprocess_files(
            preprocess_files
            ,max_length=max_length
            ,stride=stride
            ,memmap=memmap
            ,dtype=dtype
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

        train_dataset = GPTDataset.from_text(train_text, self.tokenizer, max_length, stride)
        val_dataset = GPTDataset.from_text(val_text, self.tokenizer, max_length, stride)
        
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
        assert os.path.isfile(csv_file), cvs_file
        self.data = pd.read_csv(csv_file)
        self.encoded_ids = [
            tokenizer.encode(text) for text in self.data[text_field]
        ]

        if pad_token_id is None:
            pad_token_id = tokenizer.eos_id

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

    def __getitem__(self, index: int, label_field="Label"):
        encoded_id = self.encoded_ids[index]
        label = self.data.iloc[index][label_field]
        
        return (
            torch.tensor(encoded_id, dtype=torch.long)
            , torch.tensor(label, dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self.data)

    def _longest_encoded_length(self) -> int:
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
