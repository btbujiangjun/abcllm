#
#
#
#
#

import torch
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
        max_length = max_length or self.max_length or self.model.cfg["context_length"]
        response_text = super().generate(
            start_context = start_context, 
            max_length = max_length, 
            temperature = temperature, 
            top_k = top_k, 
            eos_id = eos_id
        )
        response_text = response_text[len(start_context):].replace("\n### Response:", "").strip()
        return {
            "Start_Context":start_context,
            "Generate_Text":response_text,
        }

