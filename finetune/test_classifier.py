import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

from dataset.dataset import GPTDataset, ABCDataLoader, LabeledDataset
from tokenizer.tokenizer import GPT2Tokenizer, SimpleTokenizer
from model.pretrain_gpt2 import PretrainGPT2
from finetune.classifier import ClassifierFinetune


text = """"It's the last he painted, you know," Mrs. Gisburn Jiang said with pardonable pride."""

torch.manual_seed(123)
num_workers = 0
batch_size = 8

tokenizer = GPT2Tokenizer()

train_dataset = LabeledDataset(
    csv_file="./data/finetune/train.csv",
    max_length=None,
    tokenizer=tokenizer
)

val_dataset = LabeledDataset(
    csv_file="./data/finetune/validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

test_dataset = LabeledDataset(
    csv_file="./data/finetune/test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

train_loader = ABCDataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
    token_size = train_dataset.token_size,
)

val_loader = ABCDataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
    token_size = val_dataset.token_size,
)

test_loader = ABCDataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
    token_size = test_dataset.token_size,
)

pretrain_gpt2 = PretrainGPT2()
model = pretrain_gpt2.load_tf_ckpt("gpt2-small (124M)", "./data/pretrain_gpt2", dtype=torch.float32)
finetune = ClassifierFinetune(model, tokenizer, num_classes=2)
finetune.train(
    train_loader, 
    val_loader,
    num_epochs=2,
    eval_freq=1,
    eval_iter=1,
    start_context=text,
)

ckpt="./data/tmp/finetune/spam_finetune.ckpt"
finetune.dump(ckpt)
finetune.load(ckpt)

finetune.train(
    test_loader, 
    val_loader,
    num_epochs=2,
    eval_freq=1,
    eval_iter=1,
    start_context=text,
)

texts = (
    "You are a winner you have been specially selected to receive $1000 cash or a $2000 award.",
    "牛津剑桥名校直通车，保录取招生保专业，不录取，全额退款",
    "Hey, just wanted to check if we're still on for dinner tonight? Let me know!",
    "好好学习，天天向上!",
    "为人民服务",
    "Sorry, I'll call later",
    "Am on a train back from northampton so i'm afraid not! I'm staying skyving off today ho ho! Will be around wednesday though. Do you fancy the comedy club this week by the way?",
    "Hi, the SEXYCHAT girls are waiting for you to text them. Text now for a great night chatting. send STOP to stop this service",
    "Quite late lar... Ard 12 anyway i wun b drivin...",
    "I'm good. Have you registered to vote?",
    "Win a £1000 cash prize or a prize worth £5000"
)
for text in texts:
    predicted_label = "spam" if finetune.classifier(text, train_dataset.max_length) else "not spam"
    print(f"text:\n{text}\npredicted label is:{predicted_label}.")

