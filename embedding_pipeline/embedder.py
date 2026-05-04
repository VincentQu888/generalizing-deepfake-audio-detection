import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import warnings
from typing import Iterable
import numpy as np


class Embedder(nn.Module):
    """
    cnn-based embedding model for audio spectrograms
    """
    def __init__(
        self,
        embedding_dim: int = 128,
        dropout_rate: float = 0.2,
        conv_channels: tuple[int, int, int, int] = (16, 24, 32, 48),
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        c1, c2, c3, c4 = conv_channels
        self.final_channels = c4

        # conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.dropout = nn.Dropout(dropout_rate)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # dense layers with batch norm
        in_features = c4 * 1 * 1
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        return backbone feature map of shape (batch, channels, h, w)
        """
        x = self.conv1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        return dense embedding map via the original mlp head applied to sliding pooled patches
        """
        features = self.forward_features(x) # [B, C, Hf, Wf]
        n, _, h, w = features.shape

        patches = F.unfold(features, kernel_size=(1, 1), stride=1) # [B, C*1*1, Hf*Wf]
        num_locations = h * w

        patch_vectors = patches.transpose(1, 2).reshape(n * num_locations, -1) # [B*Hf*Wf, C]
        embeddings = self.fc2(self.fc1(patch_vectors)) # [B*Hf*Wf, embedding_dim]
        
        embeddings = F.normalize(embeddings, p=2, dim=1) # L2 normalize
        embeddings = embeddings.view(n, h*w, self.embedding_dim) # [B, Hf*Wf, embedding_dim]
        
        return embeddings

class ContrastiveLoss(nn.Module):
    """
    NT-Xent loss for contrastive learning
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, N, D) — B images, N patch embeddings each, L2 normalized
        labels:     (B, 1, 1)      — class label per image
        """
        B, N, D = embeddings.shape

        # flatten to (B*N, D)
        embeddings = embeddings.view(B * N, D)

        # each image's label applies to all N of its patches → (B*N,)
        labels = labels.squeeze().unsqueeze(1).expand(B, N).reshape(B * N)

        BN = embeddings.size(0)

        sim = torch.mm(embeddings, embeddings.t()) / self.temperature

        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.t()).float()
        mask_pos.fill_diagonal_(0)

        exp_sim = torch.exp(sim)

        mask_self = torch.eye(BN, device=embeddings.device)
        denom = (exp_sim * (1 - mask_self)).sum(dim=1, keepdim=True)

        log_prob = sim - torch.log(denom)

        n_pos = mask_pos.sum(dim=1)
        loss = -(mask_pos * log_prob).sum(dim=1) / n_pos.clamp(min=1)

        return loss[n_pos > 0].mean()