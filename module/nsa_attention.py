# nsa_attention.py
# Author: Adapted for Native Sparse Attention (NSA) based on DeepSeek's concept
# Date: 2025.06.25
# Description: PyTorch implementation of Native Sparse Attention (NSA) for efficient long-context processing.

import torch
import torch.nn as nn
import torch.nn.functional as F


class NSAAttention(nn.Module):
    """
    Native Sparse Attention (NSA) mechanism for efficient long-context processing.
    Combines token compression, selective attention, and sliding window processing
    to reduce computational complexity while preserving global and local context.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        compress_block_size: int = 4,
        compress_block_sliding_stride: int = 2,
        selection_block_size: int = 4,
        num_selected_blocks: int = 2,
        sliding_window_size: int = 2,
    ):
        """
        Initializes the NSA attention module.

        Args:
            d_model (int): Dimension of the input embeddings.
            num_heads (int): Number of attention heads.
            compress_block_size (int): Size of blocks for token compression.
            compress_block_sliding_stride (int): Stride for sliding compression blocks.
            selection_block_size (int): Size of blocks for selective attention.
            num_selected_blocks (int): Number of blocks to select for attention.
            sliding_window_size (int): Size of the sliding window for local context.

        Raises:
            ValueError: If d_model is not divisible by num_heads.
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.compress_block_size = compress_block_size
        self.compress_block_sliding_stride = compress_block_sliding_stride
        self.selection_block_size = selection_block_size
        self.num_selected_blocks = num_selected_blocks
        self.sliding_window_size = sliding_window_size

        # Linear layers for query, key, value projections
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        # Compression MLP for summarizing token blocks
        self.compress_mlp = nn.Sequential(
            nn.Linear(d_model * compress_block_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # Importance scoring layer for block selection
        self.score_layer = nn.Linear(d_model, 1)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def _compress_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compresses token sequences into block-level representations using an MLP.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Compressed representations (batch_size, num_blocks, d_model).
        """
        batch_size, seq_len, _ = x.size()
        num_blocks = (seq_len - self.compress_block_size) // self.compress_block_sliding_stride + 1

        # Reshape input into overlapping blocks
        blocks = []
        for i in range(0, seq_len - self.compress_block_size + 1, self.compress_block_sliding_stride):
            block = x[:, i:i + self.compress_block_size, :].reshape(batch_size, -1)
            blocks.append(block)
        blocks = torch.stack(blocks, dim=1)  # (batch_size, num_blocks, compress_block_size * d_model)

        # Apply MLP to compress blocks
        compressed = self.compress_mlp(blocks)  # (batch_size, num_blocks, d_model)
        return compressed

    def _select_blocks(self, compressed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Selects the most relevant blocks based on importance scores.

        Args:
            compressed (torch.Tensor): Compressed representations (batch_size, num_blocks, d_model).

        Returns:
            tuple: (selected_blocks, selected_indices)
                - selected_blocks (torch.Tensor): Selected block representations.
                - selected_indices (torch.Tensor): Indices of selected blocks.
        """
        batch_size, num_blocks, _ = compressed.size()

        # Compute importance scores for each block
        scores = self.score_layer(compressed).squeeze(-1)  # (batch_size, num_blocks)
        _, top_indices = scores.topk(self.num_selected_blocks, dim=-1)  # (batch_size, num_selected_blocks)

        # Gather selected blocks
        selected_blocks = compressed.gather(
            dim=1,
            index=top_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        )  # (batch_size, num_selected_blocks, d_model)

        return selected_blocks, top_indices

    def _sliding_window_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Applies sliding window attention for local context.

        Args:
            q (torch.Tensor): Query tensor (batch_size, num_heads, seq_len, d_head).
            k (torch.Tensor): Key tensor (batch_size, num_heads, seq_len, d_head).
            v (torch.Tensor): Value tensor (batch_size, num_heads, seq_len, d_head).

        Returns:
            torch.Tensor: Sliding window attention output (batch_size, num_heads, seq_len, d_head).
        """
        batch_size, num_heads, seq_len, d_head = q.size()
        output = torch.zeros_like(q)

        for i in range(seq_len):
            start = max(0, i - self.sliding_window_size)
            end = min(seq_len, i + self.sliding_window_size + 1)
            q_slice = q[:, :, i:i+1, :]  # (batch_size, num_heads, 1, d_head)
            k_slice = k[:, :, start:end, :]  # (batch_size, num_heads, window_size, d_head)
            v_slice = v[:, :, start:end, :]  # (batch_size, num_heads, window_size, d_head)

            scores = torch.matmul(q_slice, k_slice.transpose(-2, -1)) / (self.d_head ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            output[:, :, i:i+1, :] = torch.matmul(attn_weights, v_slice)

        return output

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes NSA attention.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Attention mask (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.size()

        # Step 1: Project input to queries, keys, and values
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (batch_size, num_heads, seq_len, d_head)

        # Step 2: Compress tokens into block representations
        compressed = self._compress_tokens(x)  # (batch_size, num_blocks, d_model)

        # Step 3: Select top-k blocks based on importance scores
        selected_blocks, _ = self._select_blocks(compressed)  # (batch_size, num_selected_blocks, d_model)

        # Step 4: Project selected blocks to keys and values for global attention
        selected_qkv = self.qkv_proj(selected_blocks).reshape(
            batch_size, self.num_selected_blocks, 3, self.num_heads, self.d_head
        )
        selected_qkv = selected_qkv.permute(2, 0, 3, 1, 4)
        selected_k, selected_v = selected_qkv[1], selected_qkv[2]  # (batch_size, num_heads, num_selected_blocks, d_head)

        # Step 5: Sliding window attention for local context
        local_output = self._sliding_window_attention(q, k, v)  # (batch_size, num_heads, seq_len, d_head)

        # Step 6: Global attention on selected blocks
        global_scores = torch.matmul(q, selected_k.transpose(-2, -1)) / (self.d_head ** 0.5)
        if mask is not None:
            global_scores = global_scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        global_attn_weights = F.softmax(global_scores, dim=-1)
        global_output = torch.matmul(global_attn_weights, selected_v)  # (batch_size, num_heads, seq_len, d_head)

        # Step 7: Combine local and global outputs
        output = local_output + global_output  # (batch_size, num_heads, seq_len, d_head)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Step 8: Final output projection
        output = self.out_proj(output)  # (batch_size, seq_len, d_model)
        return output


# Example usage and test
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Sample data
    batch_size, seq_len, d_model = 2, 32, 64
    num_heads = 8
    x = torch.randn(batch_size, seq_len, d_model)

    # Initialize NSA attention
    nsa = NSAAttention(
        d_model=d_model,
        num_heads=num_heads,
        compress_block_size=4,
        compress_block_sliding_stride=2,
        selection_block_size=4,
        num_selected_blocks=2,
        sliding_window_size=2,
    )

    # Compute output
    output = nsa(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Verify output shape
    assert output.shape == x.shape, "Output shape should match input shape"
