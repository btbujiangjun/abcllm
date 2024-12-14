import argparse
import os
from pathlib import Path
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from model.model import GPT_CONFIG_124M,GPTModel
from tokenizer.tokenizer import SPTokenizer
from dataset.dataset import GPTDataLoader


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


def _batch_loss(model, input_batch, target_batch):
    input_batch = input_batch.to(model.device)
    target_batch = target_batch.to(model.device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def _loader_loss(model, data_loader, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = _batch_loss(model, input_batch, target_batch)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def _evaluate(model, train_loader, val_loader, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = _loader_loss(model, train_loader, num_batches=eval_iter)
        val_loss = _loader_loss(model, val_loader, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def _generate(model, tokenizer, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = _text_to_token_ids(start_context, tokenizer).to(model.device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size)
        decoded_text = _token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def _text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension
    return encoded_tensor


def _token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # Remove batch dimension
    return tokenizer.decode(flat.tolist())

def train_model_simple(model,
        preprocess_train_files,
        preprocess_val_files,
        n_epochs,
        eval_freq, 
        eval_iter, 
        print_sample_iter, 
        start_context,
        output_dir, 
        save_ckpt_freq, 
        tokenizer,
        batch_size=1024 
    ):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    start_time = time.time()
    dataloader = GPTDataLoader(tokenizer, num_workers=0)

    train_loader = dataloader.preprocess_file_dataloader(
        preprocess_train_files,
        batch_size=batch_size,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"]
    )
    val_loader = dataloader.preprocess_file_dataloader(
        preprocess_val_files,
        batch_size=batch_size,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"]
    )

    try:
        for epoch in range(n_epochs):
            book_start_time = time.time()
            print("Training ...")
            model.train()
            for input_batch, target_batch in train_loader:
                model.optimizer.zero_grad()
                loss = _batch_loss(model, input_batch, target_batch)
                loss.backward()
                model.optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = _evaluate(
                        model, 
                        train_loader, 
                        val_loader, 
                        eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step}): Train tokens {tokens_seen}, "
                            f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                # Generate text passage
                if global_step % print_sample_iter == 0:
                    _generate(model, tokenizer, start_context)

                if global_step % save_ckpt_freq == 0:
                    file_name = output_dir / f"model_bk_{global_step}.pth"
                    torch.save(model.state_dict(), file_name)
                    print(f"Saved {file_name}")


    except KeyboardInterrupt:
        file_name = output_dir / f"model_pg_{global_step}_interrupted.pth"
        torch.save(model.state_dict(), file_name)
        print(f"Saved {file_name}")

    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GPT Model Training Configuration')

    parser.add_argument('--train_data', type=str, default='data/pretrain_train_data.bin',
                        help='Directory containing the training data')
    parser.add_argument('--val_data', type=str, default='data/pretrain_val_data.bin',
                        help='Directory containing the validate data')
    parser.add_argument('--output_dir', type=str, default='baidubaike_checkpoints',
                        help='Directory where the model checkpoints will be saved')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs to train the model')
    parser.add_argument('--print_sample_iter', type=int, default=1000,
                        help='Iterations between printing sample outputs')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Frequency of evaluations during training')
    parser.add_argument('--save_ckpt_freq', type=int, default=1000,
                        help='Frequency of saving model checkpoints during training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Uses a very small model for debugging purposes')

    args = parser.parse_args()


    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = SPTokenizer("tokenizer/ChatGLMTokenizer/tokenizer.model")

    train_data = args.train_data
    val_data = args.val_data
    #all_files = [os.path.join(path, name) for path, subdirs, files
    #             in os.walk(data_dir) for name in files if name.endswith((".txt"))]


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        [train_data],
        [val_data],
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        eval_iter=1,
        print_sample_iter=args.print_sample_iter,
        output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq,
        start_context="我爱北京",
        tokenizer=tokenizer
    )

    epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, output_dir)

    torch.save(model.state_dict(), output_dir / "model_pg_final.pth")
    #print(f"Maximum GPU memory allocated: {torch.mps.max_memory_allocated() / 1e9:.2f} GB")
