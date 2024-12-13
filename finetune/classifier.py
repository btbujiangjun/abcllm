#
# 
#
#
#

import torch
from model.model import GPTModel

class ClassifierFinetune():
    def __init__(self, model, tokenizer, num_classes):
        self.tokenizer = tokenizer
        self.num_classes = num_classes
        self._modify_model(model, num_classes)

    def _modify_model(self, model, num_classes):
        for param in model.parameters():
            param.requires_grad = False

        #modify out_head(emb_dim) to classification_out
        #but addtional layer can improve the performance
        model.out_head = torch.nn.Linear(
            in_features=model.cfg["emb_dim"]
            , out_features=num_classes
        ).to(model.device)

        #making last transformer block and final LayerNorm trainable
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
       
        model.cfg.update({"num_classes":num_classes})
        model.reset_optimizer()
        
        self.model = model

    def finetune(self
            ,train_loader
            ,val_loader
            ,num_epochs=1
            ,eval_freq=5
            ,eval_iter=5):
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        examples_seen, global_step = 0, 1

        for epoch in range(num_epochs):
            self.model.train()

            for input_batch, target_batch in train_loader:
                self.model.optimizer.zero_grad()
                loss = self.batch_loss(input_batch, target_batch)
                loss.backward()
                self.model.optimizer.step()


                examples_seen += input_batch.shape[0]
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate(
                        train_loader
                        ,val_loader
                        ,eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    print(f"Epoch {epoch+1} Step {global_step:06d}:"
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                    )

            train_accuracy = self.loader_accuracy(train_loader, num_batches=eval_iter)
            val_accuracy = self.loader_accuracy(val_loader, num_batches=eval_iter)
            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)
            print(f"Training accuracy:{train_accuracy*100:.2f}% |", end="")
            print(f"Validation accuracy:{val_accuracy*100:.2f}%")

        return train_losses, val_losses, train_accs, val_accs, examples_seen

    def loader_accuracy(self, data_loader, num_batches=None):
        correct_predictions, num_examples = 0, 0
        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(len(data_loader), num_batches)
        self.model.eval()


        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break

            input_batch = input_batch.to(self.model.device)
            target_batch = target_batch.to(self.model.device)

            with torch.no_grad():
                logits = self.model(input_batch)[:, -1, :] #logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()

        return correct_predictions / num_examples

    def batch_loss(self, input_batch, target_batch):
        input_batch = input_batch.to(self.model.device)
        target_batch = target_batch.to(self.model.device)
        logits = self.model(input_batch)[:, -1, :]
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
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

    def classifier(self
            ,text
            ,max_length=None
            ,pad_token_id=None):
        input_ids = self.tokenizer.encode(text)
        
        if max_length is None:
            max_length = self.model.pos_emb.weight.shape[0]
        else:
            max_length = min(max_length, self.model.pos_emb.weight.shape[0])
        
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_id()

        #truncate if too long
        input_ids = input_ids[:max_length]
        #pad if too short
        input_ids += [pad_token_id] * (max_length - len(input_ids))
        input_tensor = torch.tensor(input_ids, device=self.model.device).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)[:, -1, :]
        predicted_label = torch.argmax(logits, dim=-1).item()

        return predicted_label

    def dump(self, ckpt="classifier_finetune.pth"):
        torch.save({
            "model_cfg": self.model.cfg
            ,"model_state_dict": self.model.state_dict()
            ,"optimizer_state_dict": self.model.optimizer.state_dict()
            }, ckpt
        )

        print(f"save {ckpt} successfully.")

    def load(self, ckpt, dtype=torch.bfloat16):
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True) 
        model_cfg = checkpoint["model_cfg"]
        if self.model.cfg != model_cfg:
            self.model = GPTModel(model_cfg)
        self._modify_model(self.model, model_cfg["num_classes"])
        
        model_device = self.model.device
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in checkpoint["model_state_dict"]:
                    param.copy_(checkpoint["model_state_dict"][name].to(model_device))
                else:
                    print(f"Warning: {name} not found in state_dict.")

        if dtype is not None:
            self.model.to(dtype)
        self.model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"load {ckpt} successfully.")


