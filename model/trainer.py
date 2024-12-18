# -*- coding: utf-8 -*-
"""
trainer.py

Trainer class for training and evaluating GPT-based models. This script provides functionalities for 
training loops, evaluation, checkpoint management, and sample text generation.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: JiangJun
Date: 2024-12-16
"""

import time
import torch
import torch.nn as nn
from model.model import GPTModel, GPT_CONFIG_124M, ModelWrapper

class Trainer():
    """
    Trainer class for training and evaluating a GPTModel.

    Attributes:
        model: The GPTModel.
        tokenizer: Tokenizer used for text processing.
        wrapper (ModelWrapper): Wrapper for handling generation tasks.
        num_epochs (int): Tracks the total number of completed epochs.
        global_step (int): Global training step counter.
    """
    def __init__(self, model, tokenizer):
        """
        Initialize the Trainer object.

        Args:
            model: GPTModel to be trained.
            tokenizer: Tokenizer instance for text processing.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.wrapper = ModelWrapper()
        self.num_epochs = 0
        self.global_step = 0

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
            train_losses (list): List of training losses.
            val_losses (list): List of validation losses.
            track_tokens_seen (list): List of tokens seen at each evaluation step.
        """
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, self.global_step = 0, -1
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.model.train()

            for input_batch, target_batch in train_loader:
                self.model.optimizer.zero_grad()
                loss = self.__batch_loss(input_batch, target_batch)
                loss.backward()
                self.model.optimizer.step()
                tokens_seen += input_batch.numel()
                self.global_step += 1

                # Evaluate and log progress
                if self.global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate(train_loader, val_loader, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)

                    print(
                        f"Epoch {epoch + 1} Step {self.global_step}, Tokens_seen:{tokens_seen}, "
                        f"{tokens_seen/1000/(time.time() - start_time):.2f}k tokens/sec, "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                    )

                # Save checkpoint
                if self.global_step % dump_steps == 0:
                    self.dump(f"{dump_path}/tmp_steps_{self.global_step}.ckpt")

                # Generate sample text
                if self.global_step % sample_iter == 0:
                    generate_text = self.wrapper.generate(
                        self.model, 
                        start_context, 
                        self.tokenizer, 
                        self.model.cfg["context_length"],
                        temperature=temperature,
                        top_k=top_k,
                        eos_id=eos_id
                    )
                    print(f"Generated text:generate_text}")
       
            self.num_epochs += 1
        
        return train_losses, val_losses, track_tokens_seen


    def __batch_loss(self, input_batch, target_batch):
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
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten().long())
        return loss

    def __loader_loss(self, data_loader, num_batches=None):
        """
        Calculate the average loss over a DataLoader.

        Args:
            data_loader: DataLoader object.
            num_batches (int, optional): Number of batches to evaluate.

        Returns:
            float: Average loss.
        """
        if len(data_loader) == 0:
            return float("nan")
        if num_batches == None:
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
        """
        Evaluate the model on training and validation data.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            eval_iter (int): Number of batches to evaluate.

        Returns:
            tuple: Training loss and validation loss.
        """
        self.model.eval()
        with torch.no_grad():
            train_loss = self.__loader_loss(train_loader, num_batches=eval_iter)
            val_loss = self.__loader_loss(val_loader, num_batches=eval_iter)
        self.model.train()
        return train_loss, val_loss

    def dump(self, ckpt:str):
        """
        Save model checkpoint.

        Args:
            ckpt (str): Path to save the checkpoint.
        """
        torch.save({
            "model_cfg": self.model.cfg
            ,"num_epochs": self.num_epochs
            ,"global_step": self.global_step
            ,"model_state_dict": self.model.state_dict()
            ,"optimizer_state_dict": self.model.optimizer.state_dict()
            }, ckpt
        )
        print(f"dump ckpt {ckpt} successfully.")

    def load(self, ckpti:str, dtype=torch.bfloat16):
        """
        Load model checkpoint.

        Args:
            ckpt (str): Path to the checkpoint file.
            dtype (torch.dtype, optional): Data type to convert model parameters to.
        """
        checkpoint = torch.load(ckpt, weights_only=False, map_location="cpu")
        if self.model.cfg != checkpoint["model_cfg"]:       
            self.model = GPTModel(checkpoint["model_cfg"])
        
        self.num_epochs = checkpoint["num_epochs"] 
        self.global_step = checkpoint["global_step"]

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
        print(f"Checkpoint {ckpt} loaded with {self.num_epochs} epochs and step {self.global_step}.")
