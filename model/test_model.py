import os
import torch

from dataset.dataset import GPTDataset, GPTDataLoader
from tokenizer.tokenizer import GPT2Tokenizer
from model.gpt import GPTModel
from model.trainer import Trainer
from model.generator import Generator
from model.pretrain_gpt2 import PretrainGPT2


def initialize_model(config_name: str, ckpt_dir: str) -> GPTModel:
    """Initialize the GPT model with pre-trained weights."""
    pretrain_gpt2 = PretrainGPT2()
    model = pretrain_gpt2.load_tf_ckpt(config_name, ckpt_dir).to(torch.bfloat16)
    print(f"Model initialized with {config_name}")
    print(f"Model size (MB): {model.size_mb}")
    print(f"Number of parameters: {model.param_size}")
    return model


def prepare_dataloader(file_path: str, tokenizer, max_length: int = 8, stride: int = 8, train_ratio: float = 0.9):
    """Prepare train and validation data loaders."""
    loader = GPTDataLoader(tokenizer)
    train_loader, val_loader = loader.file_train_val_dataloader(
        file_path, train_ratio=train_ratio, seq_len=max_length, stride=stride
    )
    print(f"Data Loaders prepared: {len(train_loader)} train batches, {len(val_loader)} val batches")
    return train_loader, val_loader


def test_generation(model, tokenizer):
    """Test text generation with the model."""
    test_output = Generator.generate(model, "Who am I?", tokenizer, max_length=50)
    print("Test generation output:")
    print(test_output)


def train_and_save_model(model, tokenizer, train_loader, val_loader, dump_path: str):
    """Train the model and save it to a checkpoint."""
    trainer = Trainer(model, tokenizer)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
    )
    trainer.dump(dump_path)
    print(f"Model checkpoint saved to {dump_path}")


def load_and_continue_training(model, tokenizer, train_loader, val_loader, dump_path: str):
    """Load a checkpoint and continue training."""
    trainer = Trainer(model, tokenizer)
    trainer.load(dump_path)
    print("Checkpoint loaded. Continuing training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        eval_freq=5,
        eval_iter=5,
        start_context="miss you",
    )
    trainer.dump(dump_path)
    print(f"Model checkpoint updated and saved to {dump_path}")


def main():
    """Main function to run the entire process."""
    # File and model paths
    data_file = "./data/the-verdict.txt"
    ckpt_dir = "./data/pretrain_gpt2"
    dump_path = "./data/tmp/model/gpt2_test.ckpt"
    config_name = "gpt2-small (124M)"

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer()
    model = initialize_model(config_name, ckpt_dir)

    # Prepare data loaders
    train_loader, val_loader = prepare_dataloader(data_file, tokenizer)

    # Test text generation
    test_generation(model, tokenizer)

    # Train and save the model
    train_and_save_model(model, tokenizer, train_loader, val_loader, dump_path)

    # Continue training after loading checkpoint
    load_and_continue_training(model, tokenizer, train_loader, val_loader, dump_path)


if __name__ == "__main__":
    main()
