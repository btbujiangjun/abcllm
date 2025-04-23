import os
import sys
import json
import torch
from dataset.dataset import InstructDataset, ABCDataLoader
from tokenizer.tokenizer import SPTokenizer
from model.gpt import GPTModel
from sft.instruct import InstructTrainer


torch.manual_seed(123)
num_workers = 0
batch_size = 8
tokenizer = SPTokenizer("./data/ChatGLMTokenizer/tokenizer.model")
ignore_index = -200
#train_file = "./data/finetune/train-instruction-data.json"
#val_file = "./data/finetune/val-instruction-data.json"

#train_file = "../corpus/BAAI_COIG/human_value_alignment_instructions_part1.json"
train_file = "./data/finetune/huanhuan.json"
#train_file = "./data/finetune/west_journey.json"
val_file = "./data/finetune/val-huanhuan.json"

base_model = "data/baike_steps_171000"
model = GPTModel.from_pretrained(base_model)
model.cfg["accumulation_steps"] = 1

finetune = InstructionFinetune(model, tokenizer, max_length=128)
finetune.ignore_index = ignore_index

output_ckpt="./data/tmp/finetune/instruct"
if os.path.isdir(output_ckpt):
    finetune.load_lastest(output_ckpt)

train_dataset = InstructDataset(
    train_file, 
    tokenizer,
    seq_len=finetune.model.cfg["context_length"],
    ignore_index=ignore_index
)
train_loader = ABCDataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

val_dataset = InstructDataset(
    val_file,
    tokenizer,
    seq_len=finetune.model.cfg["context_length"],
    ignore_index=ignore_index
)
val_loader = ABCDataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

with open(val_file, "r", encoding="utf-8") as f:
    items = json.load(f)

"""
finetune.train(
    train_loader=train_loader, 
    val_loader=val_loader,
    max_length=512,
    num_epochs=1,
    eval_freq=100,
    eval_iter=1,
    sample_iter=10000,
    dump_path=output_ckpt,
    start_context=InstructDataset.format_input(items[0], with_output=True)
)
"""
with open(train_file, "r", encoding="utf-8") as f:
    items = json.load(f)

for item in items:
    data = InstructDataset.format_input(item, with_output=False)
    response_json = finetune.generate(
        start_context=data,
        temperature=0.0,
        top_k=3,
        max_length=512)
    
    print(f"Start Context:\n{response_json['Start_Context']}\n{'*' * 80}")
    print(f"\nCorrect response:>>\n {item['output']}\n{'*' * 80}")
    print(f"\nModel response:>>\n {response_json['Generate_Text']}\n{'-'*80}")

