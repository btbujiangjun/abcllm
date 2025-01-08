# -*- coding: utf-8 -*-
"""
trainer.py

Trainer class for training and evaluating GPT-based models. This script provides functionalities for 
training loops, evaluation, checkpoint management, and sample text generation.

Author: JiangJun
Date: 2024-12-16
"""
import os
import time
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from model.abcmodel import CONFIG_OPERATION
from model.manager import ModelManager
from model.generator import Generator
from module.scheduler import LinearWarmupLinearDecayScheduler

class Trainer():
    """
    Trainer class for training and evaluating a Model.

    Attributes:
        model: The Model.
        tokenizer: Tokenizer used for text processing.
        wrapper (ModelWrapper): Wrapper for handling generation tasks.
        num_epochs (int): Tracks the total number of completed epochs.
        global_step (int): Global training step counter.
    """
    def __init__(self, model, tokenizer, scheduler=None, rank=0):
        """
        Initialize the Trainer object.

        Args:
            model: Model to be trained.
            tokenizer: Tokenizer instance for text processing.
        """
        self._model = None
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.rank = rank
        self.num_epochs = 0
        self.global_step = -1
        self.manager = ModelManager(self.model)

    @property
    def model(self):
        if isinstance(self._model, DDP):
            return self._model.module
        else:
            return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def init_scheduler(self, max_steps):
        self.scheduler = LinearWarmupLinearDecayScheduler(
            self.model.optimizer,
            self.model.cfg["warmup_steps"],
            max_steps
    )


    def loss_function(self, logits, target):
        return nn.functional.cross_entropy(
            logits.flatten(0, 1),
            target.flatten().long()
        )

    def train(self
            ,train_loader
            ,val_loader
            ,num_epochs=1
            ,eval_freq=5
            ,eval_iter=5
            ,start_context=None
            ,max_length=None
            ,sample_iter=10_000
            ,dump_path="./"
            ,dump_steps=10_000
            ,dump_optimizer=False
            ,is_warmup=True
            ,temperature=0.0
            ,top_k=None
            ,eos_id=None
            ,rank=0):
        """
        Train the model and periodically evaluate and save checkpoints.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            num_epochs (int): Number of epochs to train.
            eval_freq (int): Steps between evaluations.
            eval_iter (int): Number of batches to use for evaluation.
            start_context (str): Context for text generation.
            sample_iter (int): Steps between text generation samples.
            dump_path (str): Path to save checkpoints.
            dump_steps (int): Steps between saving checkpoints.
            temperature (float): Sampling temperature for text generation.
            top_k (int): Top-k sampling for text generation.
            eos_id (int): End-of-sequence token ID.

        Returns:
            train_losses (list): List of total training losses.
            local_losses (list): List of local training losses.
            val_losses (list): List of validation losses.
            track_tokens_seen (list): List of tokens seen at each evaluation step.
        """
        if is_warmup and dump_path != r"./":
            self._load_lastest(dump_path, load_optimizer=dump_optimizer)

        if self.scheduler is None:
            self.init_scheduler(len(train_loader))

        accumulation_steps = self.model.cfg["accumulation_steps"]
        max_grad_norm = self.model.cfg["max_grad_norm"]
        if max_length is None:
            max_length = self.model.cfg["context_length"]
        
        train_losses, local_losses, val_losses, track_tokens_seen = [], [], [], []
        train_loss, local_loss = 0, 0
        tokens_seen, total_tokens = 0, num_epochs * train_loader.token_size
        local_step, local_total_step = 0, 0
        samples_seen, total_samples = 0, num_epochs * train_loader.batch_size * len(train_loader)
        start_time = time.time()

        for epoch in range(num_epochs):
            self.model.train()
            self.model.optimizer.zero_grad()
            for i, (input_batch, target_batch) in enumerate(train_loader):
                #Scale loss for gradient accumulation
                loss = self._compute_loss(input_batch, target_batch) / accumulation_steps
                loss.backward() #gridient backward
                
                local_loss += loss.item()
                tokens_seen += input_batch.numel()
                samples_seen += input_batch.shape[0]

                #Update parameters after accumulating gradients
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                    self.model.optimizer.step() #Update parameter
                    self.model.optimizer.zero_grad()
                    self.scheduler.step() #Update learning rate
                    self.global_step += 1
                    local_step += 1

                    # Evaluate and log progress
                    if self.global_step % eval_freq == 0:
                        val_loss = self.evaluate(val_loader, eval_iter)
                        
                        local_loss = local_loss / local_step
                        local_total_step += 1
                        train_loss += (local_loss - train_loss) / local_total_step 
                        delta_tokens = tokens_seen - (track_tokens_seen[-1] if track_tokens_seen else 0)
                        speed = delta_tokens / 1000 / (time.time() - start_time)
                        print(
                            f"Rank:{self.rank} Epoch:{epoch + 1} Step:{self.global_step} "
                            f"Samples seen:{samples_seen}/{total_samples} "
                            f"Tokens seen:{tokens_seen}/{total_tokens}, Speed:{speed:.2f}K tokens/sec, "
                            f"LR:{self.model.optimizer.param_groups[0]['lr']:.8f}, "
                            f"Loss(Total/Local/Val): {train_loss:.3f}/{local_loss:.3f}/{val_loss:.3f}"
                            , flush=True
                        )

                        train_losses.append(train_loss)
                        local_losses.append(local_loss)
                        val_losses.append(val_loss)
                        track_tokens_seen.append(tokens_seen)

                        local_loss, local_step = 0, 0 #refresh local matrix
                        start_time = time.time() #refresh timer

                    # Save checkpoint periodically
                    if self.rank == 0 and self.global_step > 0 and self.global_step % dump_steps == 0:
                        self.dump(
                            f"{dump_path}/tmp_epoch_{self.num_epochs}_steps_{self.global_step}", 
                            dump_optimizer=dump_optimizer
                        )
                    # Generate sample text periodically
                    if start_context is not None and self.global_step % sample_iter == 0:
                        response = self.generate(
                            start_context, 
                            max_length,
                            temperature, 
                            top_k, 
                            eos_id
                        )
                        print(f"Rank {self.rank} Generated sample:\n{response}", flush=True)
            self.num_epochs += 1

        if rank == 0:
            self.dump(
                f"{dump_path}/final_epoch_{self.num_epochs}_steps_{self.global_step}",
                dump_optimizer=dump_optimizer
            )
        
        return train_losses, local_losses, val_losses, track_tokens_seen

    def generate(self, start_context, max_length=None, temperature=0.0, top_k=None, eos_id=None):
        """
        Generate sample text from the model.

        Args:
            context (str): Starting context for generation.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling.
            eos_id (int): End-of-sequence token ID.
        """
        generate_sample = Generator.generate(
            self.model, 
            start_context, 
            self.tokenizer,
            max_length,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id
        )
        return generate_sample

    def _compute_loss(self, input_batch, target_batch):
        """
        Compute the loss for a single batch.

        Args:
            input_batch (torch.Tensor): Input data batch.
            target_batch (torch.Tensor): Target data batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        input_batch = input_batch.to(self.model.device)
        target_batch = target_batch.to(self.model.device)
        logits = self.model(input_batch)
        return self.loss_function(logits, target_batch)

    def _loader_loss(self, data_loader, num_batches=None):
        """
        Calculate the average loss over a DataLoader.

        Args:
            data_loader: DataLoader object.
            num_batches (int, optional): Number of batches to evaluate.

        Returns:
            float: Average loss.
        """    
        total_loss = 0
        num_batches = num_batches or len(data_loader)
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            total_loss += self._compute_loss(input_batch, target_batch).item()
        return total_loss / num_batches

    def evaluate(self, val_loader, eval_iter):
        self.model.eval()
        with torch.no_grad():
            val_loss = self._loader_loss(val_loader, num_batches=eval_iter)
        self.model.train()
        return val_loss

    def dump(self, ckpt:str, dump_optimizer=False):
        self.manager.dump(ckpt, 
            self.num_epochs, 
            self.global_step, 
            dump_optimizer
        )

    def load(self, ckpt:str, load_optimizer=False):
        self.num_epochs, self.global_step = self.manager.load(ckpt, load_optimizer, self.rank)

    def _load_lastest(self, ckpt_dir:str, load_optimizer=False):
        ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)]
        if len(ckpts) > 0:
            lastest_ckpt = max(ckpts, key=os.path.getmtime)
            self.load(lastest_ckpt, load_optimizer)

