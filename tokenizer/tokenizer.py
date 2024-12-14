# 
# basic modules of LLM task, preparing for embedding.
# include SimpleTokenizer and GPT2Tokenizer  
#

import os
import re
import tiktoken
from typing import List
from sentencepiece import SentencePieceProcessor 

class SimpleTokenizer:
    
    split_re = r'([,.:;?_!"()\']|--|\s)'
    sub_re = r'\s+([,.?!"()\'])'

    def __init__(self
            ,words 
            ,eot="<|endoftext|>"
            ,unk="<|unk|>"):

        words = sorted(set(words))
        words.extend([unk, eot])
        vocab = {token:integer for integer, token in enumerate(words)}

        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

        self.eot: str = eot
        self.unk: str = unk
        self.vocab_size: int = len(vocab)

    @classmethod
    def from_file(cls,
            file: str,
            encoding="utf-8",
            eot="<|endoftext|>",
            unk="<|unk|>"):

        assert os.path.isfile(file), file
        with open(file, "r", encoding=encoding) as f:
            raw_text = f.read()

        words = re.split(cls.split_re, raw_text)
        words = [item.strip() for item in words if item.strip()]

        return cls(words, eot, unk)

    def encode(self, text: str, allowed_special={"<|endoftext|>"}) -> List[int]:
        assert type(text) is str
        words = re.split(self.split_re, text)
        words = [item.strip() for item in words if item.strip()]

        words = [item if item in self.str_to_int else self.unk for item in words]
        ids = [self.str_to_int[word] for word in words]

        return ids

    def decode(self, ids: List[int]) -> str:

        text = " ".join(self.int_to_str[i] for i in ids)
        text = re.sub(self.sub_re, r'\1', text)

        return text

    @property
    def eos_id(self):
        return self.str_to_int["<|endoftext|>"]

class GPT2Tokenizer:

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text: str, allowed_special={"<|endoftext|>"}) -> List[int]:
        return self.tokenizer.encode(text, allowed_special=allowed_special)

    def decode(self, ids: List[int]) ->str:
        return self.tokenizer.decode(ids)

    @property
    def eos_id(self) -> int:
        return self.tokenizer.n_vocab - 1

class SPTokenizer:
    def __init__(self, model_file: str):
        assert os.path.isfile(model_file), model_file
        self.model = SentencePieceProcessor(model_file)

        self.bos_id: int = self.model.bos_id()
        self.eos_id: int = self.model.eos_id()
        self.vocab_size: int = self.model.vocab_size()
        
    def encode(self, text: str, bos: bool = False, eos: bool = False) ->List[int]:
        assert type(text) is str

        ids = self.model.encode(text)
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]

        return ids

    def decode(self, ids: List[int]) -> str:
        return self.model.decode(ids)

