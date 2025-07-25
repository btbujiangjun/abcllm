import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 简化的语言模型（实际中为Transformer）
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

# 奖励模型（简化为MLP）
class RewardModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)

class PPO:
    def __init__(self, model, clip_epsilon=0.2, lr=1e-4):
        self.model = model
        self.clip_epsilon = clip_epsilon
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def compute_advantage(self, rewards, values, gamma=0.99):
        advantages = []
        returns = []
        R = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            R = r + gamma * R
            advantage = R - v
            advantages.append(advantage)
            returns.append(R)
        return torch.tensor(list(reversed(advantages))), torch.tensor(list(reversed(returns)))

    def update(self, states, actions, old_log_probs, rewards, values):
        advantages, returns = self.compute_advantage(rewards, values)
        
        # 前向传播
        logits, _ = self.model(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        
        # PPO目标函数
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 值函数损失
        value_loss = (returns - values).pow(2).mean()
        
        # 总损失
        loss = policy_loss + 0.5 * value_loss
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# 训练PPO
def train_ppo():
    vocab_size = 1000
    hidden_size = 256
    model = LanguageModel(vocab_size, hidden_size)
    ppo = PPO(model)

    # 模拟数据
    states = torch.randint(0, vocab_size, (10, 20))
    actions = torch.randint(0, vocab_size, (10, 20))
    old_log_probs = torch.rand(10, 20)
    rewards = torch.rand(10, 20)
    values = torch.rand(10, 20)

    for epoch in range(100):
        loss = ppo.update(states, actions, old_log_probs, rewards, values)
        print(f"PPO Epoch {epoch}, Loss: {loss:.4f}")
