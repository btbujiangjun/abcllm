import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

from dataset.dataset import InstructionDataset, ABCDataLoader
from tokenizer.tokenizer import GPT2Tokenizer
from model.pretrain_gpt2 import PretrainGPT2
from finetune.instruction import InstructionFinetune


text = """"It's the last he painted, you know," Mrs. Gisburn Jiang said with pardonable pride."""

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
    token_size=train_dataset.token_size
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
    token_size=val_dataset.token_size
)


pretrain_gpt2 = PretrainGPT2()
model = pretrain_gpt2.load_tf_ckpt("gpt2-small (124M)", "./data/pretrain_gpt2")
finetune = InstructionFinetune(model, tokenizer, max_generate_tokens=256)
finetune.Ignore_index = ignore_index

ckpt="./data/tmp/finetune/instruct_finetune.ckpt"
if os.path.isfile(ckpt):
    finetune.load(ckpt)
finetune.train(
    train_loader, 
    val_loader,
    num_epochs=2,
    eval_freq=5,
    eval_iter=5,
    dump_path=ckpt,
    start_context=val_dataset.data[0]
)

finetune.dump(ckpt)
finetune.load(ckpt)

for data in val_dataset.data:
    input_text, response_text = finetune.generate(data)
    print(input_text)
    print(f"\nCorrect response:\n>> {data['output']}")
    print(f"\nModel response:\n>> {response_text}")
    print("-------------model response end-------------")
