import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 专家网络（前馈神经网络）
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# MoE 层
class MoELayer(nn.Module):
    def __init__(self, input_dim, num_experts, experts_per_token, hidden_dim, dropout=0.1):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, dropout) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()

        # 门控网络计算专家权重
        gate_logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-k 路由：选择每个 token 的 top-k 专家
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.experts_per_token, dim=-1)
        top_k_probs = F.softmax(top_k_probs, dim=-1)  # 重新归一化 top-k 概率

        # 初始化输出
        output = torch.zeros_like(x)

        # 对每个 token 应用 top-k 专家
        for i in range(self.experts_per_token):
            expert_idx = top_k_indices[:, :, i]  # [batch_size, seq_len]
            expert_probs = top_k_probs[:, :, i].unsqueeze(-1)  # [batch_size, seq_len, 1]

            # 收集专家输出
            expert_output = torch.zeros_like(x)
            for j in range(self.num_experts):
                mask = (expert_idx == j).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
                if mask.sum() > 0:
                    expert_input = x * mask
                    expert_output += self.experts[j](expert_input) * mask

            output += expert_output * expert_probs

        return output, gate_probs

# 负载均衡损失
def compute_load_balancing_loss(gate_probs, num_experts):
    expert_load = gate_probs.mean(dim=(0, 1))  # 每个专家的使用概率
    load_balancing_loss = num_experts * torch.var(expert_load)  # 鼓励均匀使用专家
    return load_balancing_loss

# 简化的 Transformer 模型（包含 MoE 层）
class MoETransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_experts, experts_per_token, num_heads, num_layers):
        super(MoETransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 512, d_model))  # 位置编码
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(d_model, num_heads),
                'moe': MoELayer(d_model, num_experts, experts_per_token, hidden_dim=d_model*4),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        x = self.dropout(x)

        total_load_loss = 0
        for layer in self.layers:
            # 自注意力
            attn_output, _ = layer['attention'](x, x, x)
            x = layer['norm1'](x + attn_output)

            # MoE 层
            moe_output, gate_probs = layer['moe'](x)
            x = layer['norm2'](x + moe_output)

            # 负载均衡损失
            total_load_loss += compute_load_balancing_loss(gate_probs, self.layers[0]['moe'].num_experts)

        output = self.fc_out(x)
        return output, total_load_loss

# 训练循环
def train_moe():
    # 超参数
    vocab_size = 1000
    d_model = 256
    num_experts = 8
    experts_per_token = 2
    num_heads = 8
    num_layers = 2
    batch_size = 32
    seq_len = 20
    epochs = 10

    # 初始化模型
    model = MoETransformer(vocab_size, d_model, num_experts, experts_per_token, num_heads, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 模拟数据
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 训练
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output, load_loss = model(inputs)
        task_loss = criterion(output.view(-1, vocab_size), targets.view(-1))
        total_loss = task_loss + 0.1 * load_loss  # 结合任务损失和负载均衡损失
        total_loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Task Loss: {task_loss.item():.4f}, Load Loss: {load_loss.item():.4f}")

if __name__ == "__main__":
    train_moe()
