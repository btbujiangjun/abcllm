#
# 
#
#
#

import torch
from model.model import GPTModel
from model.trainer import Trainer

class ClassifierFinetune(Trainer):
    def __init__(self, model, tokenizer, num_classes):
        self.num_classes = num_classes
        self._modify_model(model, num_classes)
        super().__init__(model, tokenizer)

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

    def loss_function(self, logits, target):
        logits = logits[:, -1, :]
        return torch.nn.functional.cross_entropy(logits, target)

    def loader_accuracy(self, data_loader, num_batches=None):
        correct_predictions, num_examples = 0, 0
        num_batches = len(data_loader) if num_batches is None else min(len(data_loader), num_batches)
        
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

    def generate(self
            ,text
            ,max_length=None
            ,temperature = 0
            ,top_k=0
            ,eos_id=None
        ):
        if max_length is None:
            max_length = self.model.pos_emb.weight.shape[0]
        else:
            max_length = min(max_length, self.model.pos_emb.weight.shape[0])
        
        if eos_id is None:
            eos_id = self.tokenizer.eos_id

        input_ids = self.tokenizer.encode(text)
        #truncate if too long
        input_ids = input_ids[:max_length]
        #pad if too short
        input_ids += [eos_id] * (max_length - len(input_ids))
        input_tensor = torch.tensor(input_ids, device=self.model.device).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)[:, -1, :]
        predicted_label = torch.argmax(logits, dim=-1).item()

        return predicted_label
    


