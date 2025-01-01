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
            max_generate_tokens=0):
        super().__init__(model, tokenizer)
        self.max_generate_tokens = max_generate_tokens
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
            start_context, 
            max_generate_tokens=None, 
            temperature=None, 
            top_k=None, 
            eos_id=None):
        input_text = InstructionDataset.format_input(start_context)
        if self.max_generate_tokens > 0:
            max_generate_tokens = self.max_generate_tokens
        response_text = super().generate(input_text, max_generate_tokens, temperature, top_k, eos_id)
        response_text = response_text[len(input_text):].replace("\n### Response:", "").strip()
        return f"Instruction Finetune:\nStart_Context:{input_text}\n{'*' * 80}\nGenerate_Text:{response_text}\n{'*' * 80}"

