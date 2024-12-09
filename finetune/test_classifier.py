import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.dataset import GPTDataset, GPTDataLoader, LabeledDataset
from tokenizer.tokenizer import GPT2Tokenizer, SimpleTokenizer
from model.pretrain_gpt2 import PretrainGPT2
from classifier import ClassifierFinetune


text = """"It's the last he painted, you know," Mrs. Gisburn Jiang said with pardonable pride."""

torch.manual_seed(123)
num_workers = 0
batch_size = 8

tokenizer = GPT2Tokenizer()

train_dataset = LabeledDataset(
    csv_file="../dataset/train.csv",
    max_length=None,
    tokenizer=tokenizer
)

val_dataset = LabeledDataset(
    csv_file="../dataset/validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

test_dataset = LabeledDataset(
    csv_file="../dataset/test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

pretrain_gpt2 = PretrainGPT2()
model = pretrain_gpt2.load_tf_ckpt("gpt2-small (124M)", "../model/gpt2")
finetune = ClassifierFinetune(model, tokenizer, num_classes=2)
#finetune.finetune(train_loader, val_loader, num_epochs=2)

ckpt="spam_finetune.ckpt"
finetune.load(ckpt)
finetune.load(ckpt)
finetune.load(ckpt)
finetune.load(ckpt)
finetune.load(ckpt)
finetune.load(ckpt)
finetune.load(ckpt)
finetune.load(ckpt)
finetune.load(ckpt)
finetune.load(ckpt)
finetune.dump(ckpt)
finetune.load(ckpt)

finetune.finetune(test_loader, val_loader, num_epochs=5)

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

