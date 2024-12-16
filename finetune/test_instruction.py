import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

from dataset.dataset import InstructionDataset, INSTRUCTION_COLLATE_FN
from tokenizer.tokenizer import GPT2Tokenizer
from model.pretrain_gpt2 import PretrainGPT2
from finetune.instruction import InstructionFinetune


text = """"It's the last he painted, you know," Mrs. Gisburn Jiang said with pardonable pride."""

torch.manual_seed(123)
num_workers = 0
batch_size = 8

tokenizer = GPT2Tokenizer()

train_dataset = InstructionDataset("./data/finetune/train-instruction-data.json", tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=INSTRUCTION_COLLATE_FN,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset("./data/finetune/val-instruction-data.json", tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=INSTRUCTION_COLLATE_FN,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)


pretrain_gpt2 = PretrainGPT2()
model = pretrain_gpt2.load_tf_ckpt("gpt2-small (124M)", "./data/pretrain_gpt2")
finetune = InstructionFinetune(model, tokenizer)

ckpt="./data/tmp/finetune/instruct_finetune.ckpt"
if os.path.isfile(ckpt):
    finetune.load(ckpt)
finetune.finetune(
    train_loader, 
    val_loader,
    num_epochs=2,
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
