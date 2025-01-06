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
from sentencepiece import SentencePieceProcessor 

class SimpleTokenizer:
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

    def encode(self, text: str) -> List[int]:
        assert isinstance(text, str), "Input text must be a string."
        words = re.split(self.split_re, text)
        words = [item.strip() for item in words if item.strip()]

        # Replace unknown words with the unknown token
        words = [item if item in self.str_to_int else self.unk for item in words]
        ids = [self.str_to_int[word] for word in words]

        return ids

    def decode(self, ids: List[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(self.sub_re, r'\1', text)# Remove unnecessary spaces

    @property
    def eos_id(self):
        """Get the ID of the end-of-text token."""
        return self.str_to_int[self.eot]
    
    @property
    def vocab_size(self) -> int:
        return len(self.int_to_str)

class GPT2Tokenizer:
    """
    Wrapper for the GPT-2 tokenizer provided by tiktoken.
    """
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids: List[int]) ->str:
        return self.tokenizer.decode(ids)

    @property
    def eos_id(self) -> int:
        """Get the ID of the end-of-text token."""
        return self.tokenizer.n_vocab - 1

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

class SPTokenizer:
    """
    SentencePiece tokenizer wrapper for encoding and decoding.
    """
    def __init__(self, tokenizer_file: str):
        assert os.path.isfile(tokenizer_file), f"The tokenizer file not found: {tokenizer_file}."
        self.tokenizer = SentencePieceProcessor(tokenizer_file)
        
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

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def eos_id(self) -> int:
        """Get the ID of the end-of-text token."""
        return self.tokenizer.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size()
