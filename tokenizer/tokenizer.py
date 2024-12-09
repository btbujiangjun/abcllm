# 
# basic modules of LLM task, preparing for embedding.
# include SimpleTokenizer and GPT2Tokenizer  
#


import re
import tiktoken


class SimpleTokenizer:
    
    split_re = r'([,.:;?_!"()\']|--|\s)'
    sub_re = r'\s+([,.?!"()\'])'

    def __init__(self
            ,words 
            ,eot="<|endoftext|>"
            ,unk="<|unk|>"):
        
        self.eot = eot
        self.unk = unk

        words = sorted(set(words))
        words.extend([unk, eot])
        vocab = {token:integer for integer, token in enumerate(words)}

        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    @classmethod
    def from_file(cls,
            file,
            encoding="utf-8",
            eot="<|endoftext|>",
            unk="<|unk|>"):

        with open(file, "r", encoding=encoding) as f:
            raw_text = f.read()

        words = re.split(cls.split_re, raw_text)
        words = [item.strip() for item in words if item.strip()]

        return cls(words, eot, unk)

    def encode(self, text, allowed_special={"<|endoftext|>"}):

        words = re.split(self.split_re, text)
        words = [item.strip() for item in words if item.strip()]

        words = [item if item in self.str_to_int else self.unk for item in words]
        ids = [self.str_to_int[word] for word in words]

        return ids

    def decode(self, ids):

        text = " ".join(self.int_to_str[i] for i in ids)
        text = re.sub(self.sub_re, r'\1', text)

        return text

    def eos_id(self):
        return self.str_to_int["<|endoftext|>"]

class GPT2Tokenizer:

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text, allowed_special={"<|endoftext|>"}):
        return self.tokenizer.encode(text, allowed_special=allowed_special)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def eos_id(self):
        return self.tokenizer.n_vocab - 1
