import torch
import torch.nn.functional as F
from model.generator import Generator
from model.manager import ModelManager

class DPO:
    def __init__(self,
            policy_model,
            ref_model,
            optimizer,
            tokenizer,
            beta=0.1):
        super().__init__()
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.loss = DPOLoss(beta)
        self.manager = ModelManager(self.policy_model)

    def batch_loss(self, batch):
        policy_chosen_logits = self.loss.log_probs(
            logits=self.policy_model(batch["chosen"]),
            labels=batch["chosen"],
            mask=batch["chosen_mask"]
        )
        policy_rejected_logits = self.loss.log_probs(
            logits=self.policy_model(batch["rejected"]),
            labels=batch["rejected"],
            mask=batch["rejected_mask"]
        )

        with torch.no_grad():
            ref_chosen_logits = self.loss.log_probs(
                logits=self.ref_model(batch["chosen"]),
                labels=batch["chosen"],
                mask=batch["chosen_mask"]
            )
            ref_rejected_logits = self.loss.log_probs(
                logits=self.ref_model(batch["rejected"]),
                labels=batch["rejected"],
                mask=batch["rejected_mask"]
            )

        loss, chosen_rewards, rejected_rewards = self.loss.dpo_loss(
            policy_chosen_logits,
            policy_rejected_logits,
            ref_chosen_logits,
            ref_rejected_logits,
        )
        return loss, chosen_rewards, rejected_rewards

    def loader_loss(self, data_loader, num_batches=None):
        if len(data_loader) == 0:
            return float("nan"), float("nan"), float("nan")

        num_batches = min(num_batches or num_batches, len(data_loader))
        total_loss, total_chosen, total_rejected = 0., 0., 0.

        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            loss, chosen, rejected = self.batch_loss(batch)
            total_loss += loss.item()
            total_chosen += chosen.item()
            total_rejected += rejected.item()

        return total_loss/num_batches, total_chosen/num_batches, total_rejected/num_batches


    def train(self,
            train_loader,
            val_loader,
            num_epochs,
            eval_freq,
            eval_iter,
            start_context,):

        tracking = {
            "train_losses": [],
            "train_chosen_rewards": [],
            "train_rejected_rewards": [],
            "val_losses": [],
            "val_chosen_rewards": [],
            "val_rejected_rewards": [],
            "tokens_seen": [],
        }
        tokens_seen, global_step = 0, -1

        for epoch in range(num_epochs):
            self.policy_model.train()
            
            for batch in train_loader:
                self.optimizer.zero_grad()
                loss, chosen_rewards, rejected_rewards = self.batch_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
                self.optimizer.step()

                tokens_seen += batch["chosen"].numel()
                global_step += 1
                
                if global_step % eval_freq == 0:
                    res = self.evaluate(train_loader, val_loader, eval_iter)
                    tracking["train_losses"].append(res["train_loss"])
                    tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                    tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                    tracking["val_losses"].append(res["val_loss"])
                    tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                    tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                    tracking["tokens_seen"].append(tokens_seen)
                    train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                    val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]

                    print(
                        f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                        f"Train reward margins {train_reward_margin:.3f}, "
                        f"Val reward margins {val_reward_margin:.3f}"
                    )

                self.generate(start_context)

        return tracking

    def evaluate(self, train_loader, val_loader, eval_iter):
        self.policy_model.eval()
        with torch.no_grad():
            train_loss, train_chosen, train_rejected = self.loader_loss(
                train_loader,
                num_batches = eval_iter
            )
            val_loss, val_chosen, val_rejected = self.loader_loss(
                val_loader,
                num_batches = eval_iter
            )
        self.policy_model.train()

        res = {
            "train_loss": train_loss,
            "train_chosen_reward": train_chosen,
            "train_rejected_reward": train_rejected,
            "val_loss": val_loss,
            "val_chosen_reward": val_chosen,
            "val_rejected_reward": val_rejected,
        }
        return res

    
    def generate(self,
            start_context:str ,
            max_length: int = None,
            context_length:int = None,
            temperature:float = 0.,
            top_k: int = None,
            eos_id: int = None
        ) -> str:
        return Generator.generate(
            model = self.policy_model,
            start_context = start_context,
            tokenizer = self.tokenizer,
            max_length = max_length,
            context_length = context_length,
            temperature = temperature,
            top_k = top_k,
            eos_id = eos_id,
        )

    def dump(self, ckpt:str, dump_optimizer=False):
        self.manager.dump(ckpt, 0, 0, dump_optimizer)


class DPOLoss:
    """
    Implements the Direct Preference Optimization (DPO) loss as described in:
    https://arxiv.org/pdf/2305.18290
    
    Loss: L_DPO = -E[log sigmoid(β(log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
    """
    def __init__(self, beta=0.1):
        self.beta = beta

    def log_probs(self, logits, labels, mask=None):
        """
        Compute log probabilities of given labels from logits

        logits:(batch_size, num_tokens, vocab_size)
        labels:(batch_size, num_tokens)
        mask:(batch_size, num_tokens)
        """
        labels = labels[:, 1:] # Shift labels to the right
        logits = logits[:, :-1, :] # Align with shifted labels
        
        log_probs = torch.gather(
            input=F.log_softmax(logits, dim=-1),
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        if mask is None:
            return log_probs.mean(dim=-1)
        else:
            return (log_probs * mask[:, 1:]).sum(dim=-1) / mask[:, 1:].sum(dim=-1)

    def dpo_loss(self,
            policy_chosen_log,
            policy_rejected_log,
            ref_chosen_log,
            ref_rejected_log):
        """ 
        Compute the DPO loss and reward margins

        xxx_log:(batch_size, num_tokens, vocab_size)
        """ 
        chosen_rewards = (policy_chosen_log - ref_chosen_log).detach()
        rejected_rewards = (policy_rejected_log - ref_rejected_log).detach()
        
        policy_log_ratio = policy_chosen_log - policy_rejected_log
        ref_log_ratio = ref_chosen_log - ref_rejected_log
        logits = policy_log_ratio - ref_log_ratio
        losses = -F.logsigmoid(self.beta * logits)

        return losses, chosen_rewards.mean(), rejected_rewards.mean()
