import argparse
import os
from pathlib import Path
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from tokenizer.tokenizer import SPTokenizer
from dataset.dataset import GPTDataLoader
from model.model import GPT_CONFIG_124M, GPTModel, ModelWrapper
from model.trainer import Trainer

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, output_dir):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(output_dir / "losses.pdf")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GPT Model Training Configuration')

    parser.add_argument('--train_data', type=str, default='./data/pretrain_train_data.bin',
                        help='Directory containing the training data')
    parser.add_argument('--val_data', type=str, default='./data/pretrain_val_data.bin',
                        help='Directory containing the validate data')
    parser.add_argument('--output_dir', type=str, default='./data/tmp/baidubaike_checkpoints',
                        help='Directory where the model checkpoints will be saved')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs to train the model')
    parser.add_argument('--print_sample_iter', type=int, default=1000,
                        help='Iterations between printing sample outputs')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Frequency of evaluations during training')
    parser.add_argument('--save_ckpt_freq', type=int, default=1000,
                        help='Frequency of saving model checkpoints during training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--warmup', type=bool, default=True,
                        help='Warmup with the lastest checkpoint')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Increase output diversity and to reduce the probability of nonsensical sentences,')
    parser.add_argument('--top_k', type=int, default=5,
                        help='restrict the sampled tokens to the top-k most likely tokens')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = SPTokenizer("./data/ChatGLMTokenizer/tokenizer.model")
    dataloader = GPTDataLoader(tokenizer, num_workers=0)
    trainer = Trainer(model, tokenizer)
    
    if args.warmup:
        ckpt_dir = args.output_dir
        ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if os.path.isfile(os.path.join(ckpt_dir, f))]
        if len(ckpts) > 0:
            lastest_ckpt = max(ckpts, key=os.path.getmtime)
            trainer.load(lastest_ckpt)

    train_loader = dataloader.preprocess_file_dataloader(
        [args.train_data],
        batch_size=args.batch_size,
        shuffle=True,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"]
    )
    val_loader = dataloader.preprocess_file_dataloader(
        [args.train_data],
        batch_size=args.batch_size,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"]
    )

    start_context = "光合作用是怎么回事"
    
    try:
        train_losses, val_losses, tokens_seen = trainer.train(
            train_loader,
            val_loader,
            num_epochs=args.num_epochs,
            eval_freq=args.eval_freq,
            eval_iter=1,
            start_context=start_context,
            sample_iter=args.print_sample_iter,
            dump_steps=args.save_ckpt_freq,
            dump_path=output_dir,
            temperature=args.temperature,
            top_k=args.top_k
        )
    
        epochs_tensor = torch.linspace(0, args.num_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, output_dir)

        trainer.dump(output_dir / "model_pg_final.pth")
    except KeyboardInterrupt:
        file_name = output_dir / "model_final_interrupted.pth"
        trainer.dump(file_name)
        print(f"Saved {file_name}")

    #print(f"Maximum GPU memory allocated: {torch.mps.max_memory_allocated() / 1e9:.2f} GB")
