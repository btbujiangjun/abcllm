# 
# trainer:
#    
#
#
import time
import torch
import torch.nn as nn
from model.model import GPTModel, GPT_CONFIG_124M, ModelWrapper

class Trainer():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.wrapper = ModelWrapper()
        self.num_epochs = 0

    def train(self
            ,train_loader
            ,val_loader
            ,num_epochs
            ,eval_freq
            ,eval_iter
            ,start_context
            ,sample_iter=10_000
            ,dump_path=""
            ,dump_steps=10_000
            ,temperature=0.0
            ,top_k=None
            ,eos_id=None):

        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.model.train()

            for input_batch, target_batch in train_loader:
                self.model.optimizer.zero_grad()
                loss = self.__batch_loss(input_batch, target_batch)
                loss.backward()
                self.model.optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate(train_loader, val_loader, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)

                    print(
                        f"Epoch {epoch+1} Step {global_step}, Tokens_seen:{tokens_seen}, "
                        f"{tokens_seen/1000/(time.time() - start_time):.2f}k tokens/sec, "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                    )

                if global_step % dump_steps == 0:
                    self.dump(f"{dump_path}/tmp_steps_{global_step}.ckpt")

                if global_step % sample_iter == 0:
                    generate_text = self.wrapper.generate(
                        self.model, 
                        start_context, 
                        self.tokenizer, 
                        self.model.cfg["context_length"],
                        temperature=temperature,
                        top_k=top_k,
                        eos_id=eos_id
                    )
                    print("generate_text:", generate_text)
       
            self.num_epochs += 1
        
        return train_losses, val_losses, track_tokens_seen


    def __batch_loss(self, input_batch, target_batch):
        input_batch = input_batch.to(self.model.device)
        target_batch = target_batch.to(self.model.device)

        logits = self.model(input_batch)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten().long())
        
        return loss

    def __loader_loss(self, data_loader, num_batches=None):
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches == None:
            num_batches = len(data_loader)
        else:
            num_batches = min(len(data_loader), num_batches)
    
        total_loss = 0
    
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            total_loss += self.__batch_loss(input_batch, target_batch).item()

        return total_loss / num_batches

    def evaluate(self, train_loader, val_loader, eval_iter):
        self.model.eval()
        with torch.no_grad():
            train_loss = self.__loader_loss(train_loader, num_batches=eval_iter)
            val_loss = self.__loader_loss(val_loader, num_batches=eval_iter)
        self.model.train()

        return train_loss, val_loss

    def dump(self, ckpt):
        torch.save({
            "model_cfg": self.model.cfg
            ,"num_epochs": self.num_epochs
            ,"model_state_dict": self.model.state_dict()
            ,"optimizer_state_dict": self.model.optimizer.state_dict()
            }, ckpt
        )
        print(f"dump ckpt {ckpt} successfully.")

    def load(self, ckpt, dtype=torch.bfloat16):
        checkpoint = torch.load(ckpt, weights_only=True, map_location="cpu")
        if self.model.cfg != checkpoint["model_cfg"]:       
            self.model = GPTModel(checkpoint["model_cfg"])
        self.num_epochs = checkpoint["num_epochs"] 

        device = self.model.device
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in checkpoint["model_state_dict"]:
                    param.copy_(checkpoint["model_state_dict"][name].to(device))
                else:
                    print(f"Warning: {name} not found in state_dict.")
        
        if dtype is not None:
            self.model.to(dtype)

        self.model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"load ckpt {ckpt} with {self.num_epochs} epochs successfully.")
