#
#
#
#
#

import torch
from dataset.dataset import InstructionDataset
from model.trainer import Trainer

class InstructionFinetune(Trainer):
    def __init__(self, 
            model, 
            tokenizer, 
            max_length=0):
        super().__init__(model, tokenizer)
        self.max_length = max_length
        self.ignore_index = -999

    def ignore_index(self, value):
        self.ignore_index = value

    def loss_function(self, logits, target):
        return torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            target.flatten().long(),
            ignore_index = self.ignore_index
        )

    def generate(self, 
            start_context: str, 
            max_length: int = None, 
            temperature:float = 0.0, 
            top_k: int = None, 
            eos_id:  int = None):
        input_text = InstructionDataset.format_input(start_context)
        if self.max_length > 0:
            max_length = self.max_length
        
        response_text = super().generate(
            start_context = input_text, 
            max_length = max_length, 
            temperature = temperature, 
            top_k = top_k, 
            eos_id = eos_id
        )
        response_text = response_text[len(input_text):].replace("\n### Response:", "").strip()
        return f"Instruction Finetune:\nStart_Context:{input_text}\n{'*' * 80}\nGenerate_Text:{response_text}\n{'*' * 80}"

