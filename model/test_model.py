import os
import sys
import torch

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataset import GPTDataset, GPTDataLoader
from tokenizer.tokenizer import GPT2Tokenizer, SimpleTokenizer
from attention.attention import MultiHeadAttention
from model.model import GPTModel,GPT_CONFIG_124M, ModelWrapper
from model.trainer import Trainer
from model.pretrain_gpt2 import PretrainGPT2


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M).to(torch.bfloat16)
print(model)
print(model.param_size)
print(model.size_mb)
file = "./data/the-verdict.txt"
tokenizer = GPT2Tokenizer()
loader = GPTDataLoader(tokenizer)
dataloader = loader.file_dataloader(file)
print("num_batchs:", len(dataloader))

'''
for inputs, targets in dataloader:
    logits = model(inputs)
    print("logits:", logits)
'''

CHOOSE_MODEL = "gpt2-small (124M)"

train_loader, val_loader = loader.file_train_val_dataloader(
    file
    ,train_ratio=0.9
    ,max_length=8
    ,stride=8
)

pretrain_gpt2 = PretrainGPT2()
model = pretrain_gpt2.load_tf_ckpt(CHOOSE_MODEL, "./data/pretrain_gpt2")

mw = ModelWrapper()
print(mw.generate(model, "红烧肉怎么做", tokenizer, 50))

trainer = Trainer(model, tokenizer)

dump_path = "./data/tmp/model/gpt2_test.ckpt"

trainer.train(
    train_loader
    ,val_loader
    ,num_epochs=5
    ,eval_freq=5
    ,eval_iter=5
    ,start_context="Every effort moves you"
)
trainer.dump(dump_path)
trainer.load(dump_path)

trainer.train(
    train_loader
    ,val_loader
    ,num_epochs=5
    ,eval_freq=5
    ,eval_iter=5
    ,start_context="miss you"
)

trainer.dump(dump_path)


