import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import GPTDataset, GPTDataLoader, LabeledDataset, InstructionDataset
from tokenizer.tokenizer import GPT2Tokenizer, SimpleTokenizer

#with open("../tokenizer/the-verdict.txt", "r", encoding="utf-8") as f:
#    raw_text = f.read()

file = "../tokenizer/the-verdict.txt"
tokenizer = GPT2Tokenizer()
loader = GPTDataLoader(tokenizer)
train_dataloader, val_dataloader = loader.file_train_val_dataloader(file, train_ratio=0.8, max_length=8, stride=7)
for inputs, targets in train_dataloader:
    for input_ids, target_ids in zip(inputs.tolist(), targets.tolist()):
        print("train_inputs:", tokenizer.decode(input_ids).replace("\n", " "))
        print("train_targets:", tokenizer.decode(target_ids).replace("\n", " "))

for inputs, targets in val_dataloader:
    for input_ids, target_ids in zip(inputs.tolist(), targets.tolist()):
        print("val_inputs:", tokenizer.decode(input_ids).replace("\n", " "))
        print("val_targets:", tokenizer.decode(target_ids).replace("\n", " "))



text = """"It's the last he painted, you know," Mrs. Gisburn Jiang said with pardonable pride."""
dataloader = loader.text_dataloader(text, max_length=5, stride=2)
for inputs, targets in dataloader:
    print(f"inputs:{inputs}")
    print(f"targets:{targets}")


train_dataset = LabeledDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)

val_dataset = LabeledDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

test_dataset = LabeledDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

for text, label in test_dataset:
    print(text)
    print(label)

with open("instruction-data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
split_idx = int(len(data)* 0.9)
with open("train-instruction-data.json", "w") as file:
    json.dump(data[:split_idx], file, indent=4)
with open("val-instruction-data.json", "w") as file:
    json.dump(data[split_idx:], file, indent=4)


instruction_dataset = InstructionDataset("instruction-data.json", tokenizer)
for d in instruction_dataset.data:
    print(InstructionDataset.instruction_format_input(d))
