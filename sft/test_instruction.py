import os
import sys
import json
import torch
from dataset.dataset import InstructionDataset, ABCDataLoader
from tokenizer.tokenizer import GPT2Tokenizer
from model.pretrain_gpt2 import PretrainGPT2
from sft.instruction import InstructionFinetune


torch.manual_seed(123)
num_workers = 0
batch_size = 8
tokenizer = GPT2Tokenizer()
ignore_index = -200
train_dataset = InstructionDataset(
    "./data/finetune/train-instruction-data.json", 
    tokenizer,
    ignore_index=ignore_index
)
train_loader = ABCDataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

val_dataset = InstructionDataset(
    "./data/finetune/val-instruction-data.json", 
    tokenizer,
    ignore_index=ignore_index
)
val_loader = ABCDataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)


pretrain_gpt2 = PretrainGPT2()
model = pretrain_gpt2.load_tf_ckpt("gpt2-small (124M)", "./data/pretrain_gpt2")
finetune = InstructionFinetune(model, tokenizer, max_length=50)
finetune.ignore_index = ignore_index

ckpt="./data/tmp/finetune/instruct"
if os.path.isdir(ckpt):
    finetune.load_lastest(ckpt)

json_file = "./data/finetune/val-instruction-data.json"
with open(json_file, "r", encoding="utf-8") as f:
    items = json.load(f)

finetune.train(
    train_loader=train_loader, 
    val_loader=val_loader,
    num_epochs=1,
    eval_freq=100,
    eval_iter=1,
    sample_iter=1100,
    dump_path=ckpt,
    start_context=InstructionDataset.format_input(items[0], with_output=True)
)

for item in items:
    data = InstructionDataset.format_input(item, with_output=False)
    response_json = finetune.generate(start_context=data, max_length=50)
    print(f"Start Context:\n{response_json['Start_Context']}\n{'*' * 80}")
    print(f"\nCorrect response:>>\n {item['output']}\n{'*' * 80}")
    print(f"\nModel response:>>\n {response_json['Generate_Text']}\n{'-'*80}")
