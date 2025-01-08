# 
# Basic modules for LLM tasks, focusing on tokenization.
# Includes implementations for SimpleTokenizer, GPT2Tokenizer, and SPTokenizer.
#
# Author: Jiang Jun
# Date: 2024-12-19
#

import os
import re
import tiktoken
from typing import List
from abc import ABC, abstractmethod

class ABCTokenizer(ABC):
    def __init__(self, eos_id: int, vocab_size: int):
        self._eos_id = eos_id
        self._vocab_size = vocab_size

    @abstractmethod
    def encode(self, text:str)->List[int]:
        pass

    @abstractmethod
    def decode(self, ids:List[int])->str:
        pass

    @property
    def eos_id(self)->int:
        return self._eos_id

    @property
    def vocab_size(self)->int:
        return self._vocab_size

class SimpleTokenizer(ABCTokenizer):
    """
    A simple tokenizer for splitting text into tokens based on predefined rules.
    Provides basic encode/decode functionality for LLM tasks.
    """

    # Regex for splitting text into tokens
    split_re = r'([,.:;?_!"()\']|--|\s)'
    # Regex for post-processing decoded text
    sub_re = r'\s+([,.?!"()\'])'

    def __init__(self
            ,words 
            ,eot="<|endoftext|>"
            ,unk="<|unk|>"):
        """
        Initialize the SimpleTokenizer with a vocabulary.

        Args:
            words (list): List of words to include in the vocabulary.
            eot (str): End-of-text token (default: "<|endoftext|>").
            unk (str): Unknown token (default: "<|unk|>").
        """
        self.eot: str = eot
        self.unk: str = unk
        
        words = sorted(set(words))
        words.extend([unk, eot])

        self.str_to_int = {token:integer for integer, token in enumerate(words)}
        self.int_to_str = {i:s for s, i in self.str_to_int.items()}

        super().__init__(self.str_to_int[eot], len(self.int_to_str)) 

    @classmethod
    def from_file(cls,
            file: str,
            encoding="utf-8",
            eot="<|endoftext|>",
            unk="<|unk|>"):
        """
        Create a SimpleTokenizer from a vocabulary file.

        Args:
            file (str): Path to the vocabulary file.
            encoding (str): File encoding (default: "utf-8").
            eot (str): End-of-text token (default: "<|endoftext|>").
            unk (str): Unknown token (default: "<|unk|>").
        """
        assert os.path.isfile(file), f"File not found:{file}."
        with open(file, "r", encoding=encoding) as f:
            raw_text = f.read()

        words = re.split(cls.split_re, raw_text)
        words = [item.strip() for item in words if item.strip()]

        return cls(words, eot, unk)

    def encode(self, text: str)->List[int]:
        assert isinstance(text, str), "Input text must be a string."
        words = re.split(self.split_re, text)
        words = [item.strip() for item in words if item.strip()]

        # Replace unknown words with the unknown token
        words = [item if item in self.str_to_int else self.unk for item in words]
        ids = [self.str_to_int[word] for word in words]

        return ids

    def decode(self, ids: List[int])->str:
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(self.sub_re, r'\1', text)# Remove unnecessary spaces

class GPT2Tokenizer(ABCTokenizer):
    """
    Wrapper for the GPT-2 tokenizer provided by tiktoken.
    """
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        super().__init__(self.tokenizer.n_vocab - 1, self.tokenizer.n_vocab)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids: List[int]) ->str:
        return self.tokenizer.decode(ids)

class SPTokenizer(ABCTokenizer):
    """
    SentencePiece tokenizer wrapper for encoding and decoding.
    """
    def __init__(self, tokenizer_file: str):
        assert os.path.isfile(tokenizer_file), f"The tokenizer file not found: {tokenizer_file}."
        from sentencepiece import SentencePieceProcessor
        self.tokenizer = SentencePieceProcessor(tokenizer_file)
        super().__init__(self.tokenizer.eos_id(), self.tokenizer.vocab_size())

    def encode(self, text: str, bos: bool = False, eos: bool = False) ->List[int]:
        """
        Encode text into token IDs using SentencePiece.

        Args:
            text (str): The input text.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
        """
        assert isinstance(text, str), "Input text must be a string."

        ids = self.tokenizer.encode(text)
        if bos:
            ids = [self.tokenizer.bos_id()] + ids
        if eos:
            ids = ids + [self.tokenizer.eos_id()]

        return ids

    def decode(self, ids: List[int])->str:
        return self.tokenizer.decode(ids)

class JsonTokenizer(ABCTokenizer):
    def __init__(self, json_file:str):
        assert os.path.isfile(json_file), f"Tokenizer file {json_file} not found."
        from transformers import PreTrainedTokenizerFast
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=json_file)
        super().__init__(self.tokenizer.eos_token_id, self.tokenizer.vocab_size)

    def encode(self, text: str)->list[int]:
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, ids: List[int])->str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

