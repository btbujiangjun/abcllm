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

    def encode(self, text: str, allowed_special={"<|endoftext|>"}) -> List[int]:
        """
        Encode a string into a list of token IDs.

        Args:
            text (str): The input text.
            allowed_special (set): Special tokens allowed in the text (default: {"<|endoftext|>"}).
        """
        assert isinstance(text, str), "Input text must be a string."
        words = re.split(self.split_re, text)
        words = [item.strip() for item in words if item.strip()]

        # Replace unknown words with the unknown token
        words = [item if item in self.str_to_int else self.unk for item in words]
        ids = [self.str_to_int[word] for word in words]

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs into a string.

        Args:
            ids (List[int]): List of token IDs to decode.
        """
        text = " ".join(self.int_to_str[i] for i in ids)
        text = re.sub(self.sub_re, r'\1', text)# Remove unnecessary spaces

        return text

    @property
    def eos_id(self):
        """Get the ID of the end-of-text token."""
        return self.str_to_int["<|endoftext|>"]

class GPT2Tokenizer:
    """
    Wrapper for the GPT-2 tokenizer provided by tiktoken.
    """
    def __init__(self):
        """Initialize the GPT2Tokenizer."""
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text: str, allowed_special={"<|endoftext|>"}) -> List[int]:
        """
        Encode text into token IDs using GPT-2's tokenizer.

        Args:
            text (str): The input text.
            allowed_special (set): Allowed special tokens.
        """
        return self.tokenizer.encode(text, allowed_special=allowed_special)

    def decode(self, ids: List[int]) ->str:
        """
        Decode token IDs into text using GPT-2's tokenizer.

        Args:
            ids (List[int]): List of token IDs to decode.
        """
        return self.tokenizer.decode(ids)

    @property
    def eos_id(self) -> int:
        """Get the ID of the end-of-text token."""
        return self.tokenizer.n_vocab - 1

class SPTokenizer:
    """
    SentencePiece tokenizer wrapper for encoding and decoding.
    """
    def __init__(self, model_file: str):
        """
        Initialize the SPTokenizer with a SentencePiece model file.

        Args:
            model_file (str): Path to the SentencePiece model file.
        """
        assert os.path.isfile(model_file), f"Model file not found: {model_file}."
        self.model = SentencePieceProcessor(model_file)

        self.bos_id: int = self.model.bos_id()
        self.eos_id: int = self.model.eos_id()
        self.vocab_size: int = self.model.vocab_size()
        
    def encode(self, text: str, allowed_special={}, bos: bool = False, eos: bool = False) ->List[int]:
        """
        Encode text into token IDs using SentencePiece.

        Args:
            text (str): The input text.
            allowed_special (set): Allowed special tokens.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
        """
        assert isinstance(text, str), "Input text must be a string."

        ids = self.model.encode(text)
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs into text using SentencePiece.

        Args:
            ids (List[int]): List of token IDs to decode.
        """
        return self.model.decode(ids)

