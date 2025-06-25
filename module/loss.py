# loss.py
# Author: Jiang Jun
# Date: 2025.06.25
# Description: Implementation of various loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F 

class CrossEntropyLoss(nn.Module):
    """
    Implements Cross Entropy Loss for classification tasks.
    - Binary classification: 
        BCE = -1/n * Σ(y_i * log(ŷ_i) + (1-y_i) * log(1-ŷ_i))
    - Multi-class classification: 
        CE = -1/n * Σ_i Σ_j (y_i_j * log(ŷ_i_j))
    Suitable for classifiers and autoregressive models (e.g., GPT-like).
    """
    def __init__(self):
        super().__init__()

    def forward(self, 
            logits: torch.Tensor, 
            targets: torch.Tensor
        ) -> torch.Tensor:
        """
        logits : Raw model outputs (batch_size, num_classes).
        targets: True class indices (batch_size,).
        """
        # apply softmax and log
        log_probs = F.log_softmax(logits, dim=-1)
        # negative log-likehood NLL
        return F.nll_loss(log_probs, targets) 

class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Implements Cross Entropy Loss with label smoothing to reduce overfitting.
    Smoothes the target distribution by blending one-hot vectors with uniform distribution.
    (1 - ε) * one_hot + ε / num_classes
    """
    def __init__(self, 
            epsilon: float = 0.1, 
            reduction: str = 'mean'
        ):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, 
            logits: torch.Tensor, 
            targets: torch.Tensor
        ) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
       
        # Convert targets to one-hot if needed 
        if targets.dim() == 1:
            targets = F.one_hot(
                targets, 
                num_classes=num_classes
            ).float()

        # Apply label smoothing: (1 - ε) * one_hot + ε / num_classes
        smoothed_targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
       
        # Compute loss: -sum(smoothed_targets * log_probs) 
        loss = (-smoothed_targets * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    """
    Implements Contrastive Loss for learning embeddings by comparing pairs of samples.
    Encourages similar samples to be close and dissimilar samples to be far apart.
    
    Loss formula:
        L = y * ||x1 - x2||^2 + (1 - y) * max(margin - ||x1 - x2||, 0)^2
    where:
        - x1, x2: Embeddings of the sample pair
        - y: 1 for positive pairs (similar), 0 for negative pairs (dissimilar)
        - margin: Minimum distance for negative pairs
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, 
            x1: torch.Tensor, 
            x2: torch.Tensor, 
            label: torch.Tensor
        ) -> torch.Tensor:
        """
        x1/x2: Embedding of first/second sample.
        label: 1 for positive pairs, 0 for negative pairs.
        """        
        distance = F.pairwise_distance(x1, x2)
        loss = label * distance.pow(2) + (1 - label) * F.relu(self.margin - distance).pow(2)
        return loss.mean()

class TripletLoss(nn.Module):
    """
    Implements Triplet Loss for learning embeddings using anchor, positive, and negative samples.
    Encourages anchor-positive distance to be smaller than anchor-negative distance by a margin.
    
    Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
    where d is the distance metric (Euclidean or cosine).    
    """
    def __init__(self, 
            margin: float = 1.0, 
            distance: str = 'euclidean'
        ):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def calc_distance(self, 
            x1: torch.Tensor, 
            x2: torch.Tensor
        ) -> torch.Tensor:
        if self.distance == 'euclidean':
            return F.pairwise_distance(x1, x2)
        elif self.distance == 'cosine':
            return 1 - F.cosine_similarity(x1, x2)
        else:
            raise ValueError(f"Unsupported distance metric:{self.distance}")

    def forward(self, 
            anchor: torch.Tensor, 
            positive: torch.Tensor, 
            negative: torch.Tensor
        ) -> torch.Tensor:
        """
        anchor: Anchor embedding.
        positive: Positive embedding (similar to anchor).
        negative: Negative embedding (dissimilar to anchor).
        """
        pos_distance = self.calc_distance(anchor, positive)
        neg_distance = self.calc_distance(anchor, negative)
        loss = F.relu(pos_distance - neg_distance + self.margin)
        return loss.mean()

class FocalLoss(nn.Module):
    """
    Implements Focal Loss to focus on hard-to-classify examples.
    Modulates cross-entropy loss by down-weighting easy examples.
    """
    def __init__(self, 
            alpha: float = 0.25, 
            gamma: float = 2.0, 
            reduction: str = 'mean'
        ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, 
            inputs: torch.Tensor, 
            targets: torch.Tensor
        ) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, 
            targets, 
            reduction='none'
        )
        pt = torch.exp(-bce_loss) # Probability of true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: 
            return focal_loss

class KLdivergenceLoss(nn.Module):
    """
    Implements Kullback-Leibler Divergence Loss for comparing probability distributions.
    Measures how one distribution diverges from another.
    """
    def __init__(self):
        super().__init__()

    def forward(self, 
            logits_p: torch.Tensor, 
            logits_q: torch.Tensor
        ) -> torch.Tensor:
        """
        logits_p: Logits of the target distribution.
        logits_q: Logits of the predicted distribution.
        """
        p = F.softmax(logits_p, dim=-1)
        log_p = F.log_softmax(logits_p, dim=-1)
        log_q = F.log_softmax(logits_q, dim=-1)

        return F.kl_div(log_q, log_p, reduction='batchmean', log_target=True)


