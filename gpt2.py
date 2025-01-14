import os
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from tokenizer.tokenizer import SPTokenizer
from dataset.dataset import GPTDataset, ABCDataLoader, GPTDataLoader
from model.gpt import GPT_CONFIG_124M, GPTModel
from model.trainer import Trainer

def plot_losses(epochs_seen, tokens_seen, train_losses, local_losses, val_losses, output_dir):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, local_losses, label="Local loss")
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
    plt.savefig(f"{output_dir}/losses.pdf")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_worker(
        rank,
        world_size,
        args,
        model_cfg,
        is_distributed=False,
    ):
    is_main_processor = True
    tokenizer = SPTokenizer("./data/ChatGLMTokenizer/tokenizer.model")
    dataloader = GPTDataLoader(tokenizer, num_workers=0)
    
    if is_distributed:
        is_main_processor = (rank == 0)
        setup(rank, world_size)
        dataset = GPTDataset.from_preprocess_files(
            [args.train_data],
            max_length=model_cfg["context_length"],
            stride=model_cfg["context_length"],
            memmap=True
        )
        model_cfg['device'] = torch.device(f"cuda:{rank}")
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = ABCDataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
    else:
        train_loader = dataloader.preprocess_file_dataloader(
            [args.train_data],
            batch_size=args.batch_size,
            shuffle=True,
            max_length=model_cfg["context_length"],
            stride=model_cfg["context_length"]
        )
    
    val_loader = dataloader.preprocess_file_dataloader(
        [args.train_data],
        batch_size=args.batch_size,
        max_length=model_cfg["context_length"],
        stride=model_cfg["context_length"]
    )

    model = GPTModel(model_cfg)
    if is_distributed:
        model = DDP(model, device_ids=[rank])
    trainer = Trainer(model, tokenizer, rank=rank)

    if is_distributed:    
        torch.distributed.barrier()

    try:
        start_context = "宇宙起源"
        train_losses, local_losses, val_losses, tokens_seen = trainer.train(
            train_loader,
            val_loader,
            num_epochs=args.num_epochs,
            eval_freq=args.eval_freq,
            eval_iter=1,
            start_context=start_context,
            sample_iter=args.print_sample_iter,
            dump_steps=args.save_ckpt_freq,
            dump_path=args.output_dir,
            dump_optimizer=args.dump_optimizer,
            is_warmup=args.warmup,
            temperature=args.temperature,
            top_k=args.top_k
        )
    
        if is_main_processor:
            trainer.dump(
                f"{args.output_dir}/model_final",
                dump_optimizer=args.dump_optimizer
            )
    
        if is_distributed:
            cleanup()

        epochs_tensor = torch.linspace(0, args.num_epochs, len(train_losses))
        plot_losses(
            epochs_tensor, 
            tokens_seen, 
            train_losses, 
            local_losses,
            val_losses, 
            args.output_dir
        )

    except KeyboardInterrupt:
        if rank == 0:
            trainer.dump(
                f"{args.output_dir}/model_final_interrupted",
                dump_optimizer=args.dump_optimizer
            )

def for_server_conf(args, model_conf):
    #args.train_data = "/disk6/data/baby_data/baidubaike/baidubaike_563w_train_0.bin"
    #args.train_data = "/disk6/data/baby_data/baidubaike/baidubaike_563w_train_1.bin"
    #args.train_data = "/disk6/data/baby_data/c4_zh/c4_zh_train_0.bin"
    args.train_data = "/disk6/data/baby_data/wiki/wiki.bin"
    args.val_data = "/disk6/data/baby_data/baidubaike/baidubaike_563w_val.bin"
    args.output_dir = "/disk6/data/baidubaike_checkpoints_multi_gpu"
    args.print_sample_iter = 250
    args.eval_freq = 50
    args.save_ckpt_freq = 1000
    args.batch_size = 4
    model_conf["context_length"] = 768
    model_conf["accumulation_steps"] = 40


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
    parser.add_argument('--print_sample_iter', type=int, default=50,
                        help='Iterations between printing sample outputs')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Frequency of evaluations during training')
    parser.add_argument('--save_ckpt_freq', type=int, default=500,
                        help='Frequency of saving model checkpoints during training')
    parser.add_argument('--dump_optimizer', type=bool, default=False,
                        help='saving optimizer state during training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--warmup', type=bool, default=True,
                        help='Warmup with the lastest checkpoint')
    parser.add_argument('--temperature', type=float, default=1.5,
                        help='Increase output diversity and to reduce the probability of nonsensical sentences,')
    parser.add_argument('--top_k', type=int, default=15,
                        help='restrict the sampled tokens to the top-k most likely tokens')

    args = parser.parse_args()
    model_cfg = GPT_CONFIG_124M
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.lr:
        model_cfg["lr"] = args.lr

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        for_server_conf(args, model_cfg)
        mp.spawn(train_worker, args=(world_size, args, model_cfg, True), nprocs=world_size, join=True)
    else:
        train_worker(0, 1, args, model_cfg, False)

    print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
