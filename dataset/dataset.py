"""
File: dataset.py
Description: This module provides classes and methods for creating datasets and data loaders 
             for GPT-style language models. It includes support for text, file, and preprocessed 
             data input, as well as labeled and instruction-based datasets.
Author: Jiang Jun
Date: 2024-12-19
Dependencies: torch, pandas, numpy
"""

import os
import json
import warnings
from typing import List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    """
    A dataset class for GPT-style models. It generates input and target sequences 
    based on a sliding window approach over tokenized data.

    Args:
        token_ids (List[int]): Tokenized input data.
        seq_len (int): Maximum length of each sequence.
        stride (int): Step size for the sliding window.
    """
    def __init__(self
            ,token_ids
            ,seq_len=256
            ,stride=1
            ,eos_id=0):
        if len(token_ids) < seq_len - 1:
            raise ValueError(f"token size({len(token_ids)}) is less then seq_len({seq_len}).")
       
        self.seq_len = seq_len
        self.stride = stride
        self.eos_id = eos_id
        self.token_ids = token_ids
        self.token_size = len(token_ids)
        self.len = (self.token_size - seq_len) // stride + 1

    @classmethod
    def from_text(cls
            ,text:str
            ,tokenizer
            ,seq_len=256
            ,stride=1):
        return cls(tokenizer.encode(text), seq_len, stride, tokenizer.eos_id)

    @classmethod
    def from_files(cls
            ,files:List[str]
            ,tokenizer
            ,seq_len=256
            ,stride=1
            ,encoding="utf-8"):
        text = []
        for file in files:
            if os.path.isfile(file):
                with open(file, "r", encoding=encoding) as f:
                    text.append(f.read())
            else:
                warnings.warn(f"{file} does not existed. Skipping.")
        return cls.from_text("".join(text), tokenizer, seq_len, stride)
    
    @classmethod
    def from_preprocess_files(cls
            ,preprocess_files: List[str]
            ,seq_len=256
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
        
        return cls(token_ids, seq_len, stride)
                

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        Retrieves the input and target sequence for the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and target sequences.
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}.") 
        
        start_index = idx * self.stride
        end_index = start_index + self.seq_len + 1 #for target shift 1
        
        data_seq = np.asarray(self.token_ids[start_index : end_index], dtype=np.int32)
        padding_length = max(0, self.seq_len + 1 - len(data_seq))
        if padding_length > 0:
            data_seq = np.concatenate([data_seq, np.full(padding_length, self.eos_id)])

        input_batch = torch.from_numpy(data_seq[: -1]).to(dtype=torch.int32)
        target_batch = torch.from_numpy(data_seq[1:]).to(dtype=torch.int32)
        
        return input_batch, target_batch


class ABCDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_size = self.dataset.token_size

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
        return ABCDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
        )

    def text_dataloader(self
            ,text: str 
            ,batch_size=4
            ,seq_len=256
            ,stride=128
            ,shuffle=True
            ,drop_last=True):
        dataset = GPTDataset.from_text(text, self.tokenizer, seq_len, stride)
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
            ,seq_len=256
            ,stride=256
            ,shuffle=False
            ,drop_last=True):

        if not isinstance(file, list):
            file = [file]

        dataset = GPTDataset.from_files(
            file                        
            ,self.tokenizer
            ,seq_len
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
            ,seq_len=256
            ,stride=256
            ,shuffle=False
            ,drop_last=True
            ,memmap=True
            ,dtype="uint16"):
        
        dataset = GPTDataset.from_preprocess_files(
            preprocess_files
            ,seq_len=seq_len
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
            ,seq_len=256
            ,stride=128):
        assert train_ratio > 0.0 and train_ratio < 1.0, "train_ratio should in (0.0, 1.0)"
        split_idx = int(len(text) * train_ratio)
        train_text, val_text = text[:split_idx], text[split_idx:]

        train_loader = self.text_dataloader(train_text, batch_size, seq_len, stride, shuffle=True, drop_last=True)
        val_loader = self.text_dataloader(val_text, batch_size, seq_len, stride, shuffle=False, drop_last=False)

        return train_loader, val_loader


    def file_train_val_dataloader(self
            ,file
            ,encoding="utf-8"
            ,train_ratio=0.9
            ,batch_size=4
            ,seq_len=256
            ,stride=128):
        with open(file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        return self.text_train_val_dataloader(
            raw_text
            ,train_ratio
            ,batch_size
            ,seq_len
            ,stride
        )


class LabeledDataset(Dataset):
    def __init__(self
            ,csv_file
            ,tokenizer
            ,seq_len=None
            ,pad_token_id=None
            ,text_field="Text"
            ,label_field="Label"):
        assert os.path.isfile(csv_file), f"File {cvs_file} not found."
        data = pd.read_csv(csv_file)
        self.encoded_ids, self.labels = zip(*[(tokenizer.encode(text), label) for text, label in zip(data[text_field], data[label_field])])

        self.seq_len = seq_len or max([len(encoded_id) for encoded_id in self.encoded_ids])
        if seq_len is None:
            self.encoded_ids = [
                encoded_id[:self.seq_len] for encoded_id in self.encoded_ids
            ]

        pad_token_id = pad_token_id or tokenizer.eos_id
        self.encoded_ids = [
            encoded_id + [pad_token_id] * (self.seq_len - len(encoded_id))
            for encoded_id in self.encoded_ids
        ]
        self.token_size = self.seq_len * len(self.encoded_ids)

    def __getitem__(self, index: int):
        return (
            torch.tensor(self.encoded_ids[index], dtype=torch.long)
            , torch.tensor(self.labels[index], dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self.labels)

class InstructDataset(Dataset):
    def __init__(self, 
            json_file, 
            tokenizer, 
            seq_len=1024, 
            ignore_index=-100, 
            file_encoding="utf-8"
        ):
        assert os.path.isfile(json_file), f"File {json_file} not found."
        with open(json_file, "r", encoding=file_encoding) as f:
            data = json.load(f)
        self.encoded_ids = [tokenizer.encode(self.format_input(item, with_output=True))[:seq_len] for item in data]
            
        self.seq_len = min(seq_len, max([len(item) for item in self.encoded_ids]))
        self.tokenizer = tokenizer
        self.token_size = len(self.encoded_ids) * self.seq_len
        self.ignore_index = ignore_index

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
        item = self.encoded_ids[index].copy() + [self.tokenizer.eos_id]
        item = item + [self.tokenizer.eos_id] * (self.seq_len + 1 - len(item))
        input_tensor = torch.tensor(item[:-1]) #truncate the last token
        target_tensor = torch.tensor(item[1:]) #shift +1 to the right for targets
       
        #Don't compute loss on ignore_index
        mask = (target_tensor == self.tokenizer.eos_id)
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            target_tensor[indices[1:]] = self.ignore_index

        return input_tensor, target_tensor

    def __len__(self):
        return len(self.encoded_ids)

class PreferenceDataset(Dataset):
    def __init__(self, 
            json_file, 
            tokenizer,
            seq_len=None,
            mask_prompt=True,
            file_encoding="utf-8",
        ):
        self.tokenizer = tokenizer

        assert os.path.isfile(json_file), f"File {json_file} not found."
        with open(json_file, "r", encoding=file_encoding) as f:
            data = json.load(f)

        self.encoded_ids = []
        max_length = 0
        split_tokens = tokenizer.encode("\n\n")

        for entry in data:
            prompt = tokenizer.encode(self.format_input(entry))
            chosen = tokenizer.encode(f"### Response:\n{entry['chosen']}")
            rejected = tokenizer.encode(f"### Response:\n{entry['rejected']}")
            
            if not mask_prompt:
                chosen = prompt + split_tokens + chosen
                rejected = prompt + split_tokens + rejected

            if seq_len is not None:
                prompt = prompt[:seq_len]
                chosen = chosen[:seq_len]
                adjected = adjected[:seq_len]
            else:
                max_length = max([len(prompt), len(chosen), len(rejected), max_length])

            self.encoded_ids.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }) 
                
        self.max_length = seq_len or max_length
        self.token_size = len(self.encoded_ids) * self.max_length * 3

    @staticmethod
    def format_input(entry):
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )
        input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
        return instruction_text + input_text

    def __getitem__(self, index):
        data = self.encoded_ids[index]
        item = {} 
        
        for key in data.keys():#["chosen", "rejected"]:
            tokens = data[key]
            padded = tokens + [self.tokenizer.eos_id] * (self.max_length - len(tokens))
            item[key] = torch.tensor(padded)
            
            if key in ["chosen", "rejected"]:
                mask = torch.ones(len(padded)).bool()
                mask[len(tokens):] = False
                item[f"{key}_mask"] =  mask

        return item

    def __len__(self):
        return len(self.encoded_ids)

