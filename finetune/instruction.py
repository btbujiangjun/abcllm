#
#
#
#
#

import torch
from dataset.dataset import InstructionDataset
from model.model import GPTModel, ModelWrapper

class InstructionFinetune():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.mw = ModelWrapper()

    def finetune(self,
            train_loader,
            val_loader,
            start_context,
            num_epochs=1,
            eval_freq=5,
            eval_iter=5):
        train_losses, val_losses, track_tokens_seen = [], [], []
        token_seen, global_step = 0, -1
        
        for epoch in range(num_epochs):
            for input_batch, target_batch in train_loader:
                self.model.train()
                self.model.optimizer.zero_grad()
                loss = self.batch_loss(input_batch, target_batch)
                loss.backward()
                self.model.optimizer.step()
                token_seen += input_batch.numel()
                global_step += 1
                
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate(
                        train_loader,
                        val_loader,
                        eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(token_seen)
                    print(
                        f"Epoch {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, "
                        f"Val loss {val_loss:.3f}"
                    )
                
            input_text, generate_text = self.generate(start_context)
            print(
                f"instruction finetune:\n"
                f"start_context:{input_text}\n"
                f"generate_text:{generate_text}"
            )

        return train_losses, val_losses, track_tokens_seen

    def batch_loss(self, input_batch, target_batch):
        input_batch = input_batch.to(self.model.device())
        target_batch = target_batch.to(self.model.device())
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            target_batch.flatten()
        )

        return loss

    def loader_loss(self, data_loader, num_batches=None):
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
            total_loss += self.batch_loss(input_batch, target_batch).item()

        return total_loss / num_batches

    def evaluate(self, train_loader, val_loader, eval_iter):
        self.model.eval()
        with torch.no_grad():
            train_loss = self.loader_loss(train_loader, num_batches=eval_iter)
            val_loss = self.loader_loss(val_loader, num_batches=eval_iter)
        self.model.train()

        return train_loss, val_loss

    def generate(self, json_data, max_tokens=256):
        input_text = InstructionDataset.format_input(json_data)
        response_text = self.mw.generate(
            self.model, 
            input_text, 
            self.tokenizer, 
            max_tokens, 
            eos_id=self.tokenizer.eos_id()
        )
        return input_text, response_text[len(input_text):].replace("\n### Response:", "").strip()

    def dump(self, ckpt="instruction_finetune.pth"):
        torch.save({
            "model_cfg": self.model.cfg
            ,"model_state_dict": self.model.state_dict()
            ,"optimizer_state_dict": self.model.optimizer.state_dict()
            }, ckpt
        )

        print(f"save {ckpt} successfully.")

    def load(self, ckpt, dtype=torch.bfloat16):
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
        if self.model.cfg != checkpoint["model_cfg"]:
            self.model = GPTModel(checkpoint["model_cfg"])

        model_device = self.model.device()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                state_dict = checkpoint["model_state_dict"]
                if name in state_dict:
                    param.copy_(state_dict[name].to(model_device))
                else:
                    printf(f"Warning:{name} not found in state_dict.")
        if dtype is not None:
            self.model.to(dtype)
        self.model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"load {ckpt} successfully.")
