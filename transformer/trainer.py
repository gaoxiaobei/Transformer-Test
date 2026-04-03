"""Training utilities for Transformer."""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLRScheduler(_LRScheduler):
    """Learning rate scheduler with warmup as described in the paper.

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self._step_count
        scale = self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization."""

    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (batch, seq_len, vocab_size) - logits
        # target: (batch, seq_len)
        pred = pred.contiguous().view(-1, pred.size(-1))
        target = target.contiguous().view(-1)

        # Apply log_softmax since model outputs logits, KLDivLoss expects log_probs
        pred = torch.log_softmax(pred, dim=-1)

        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # -2 for padding and true token
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = target == self.padding_idx
        true_dist[mask] = 0

        return self.criterion(pred, true_dist)


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    device: torch.device,
    pad_idx: int,
    grad_clip_norm: float = 1.0,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in train_loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        # Teacher forcing: input is tgt[:-1], target is tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt_input != pad_idx).unsqueeze(1).unsqueeze(2) & \
                   torch.tril(torch.ones(tgt_input.size(1), tgt_input.size(1), device=device)).bool()

        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(output, tgt_output)
        loss.backward()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()
        scheduler.step()

        n_tokens = (tgt_output != pad_idx).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens, 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    pad_idx: int,
) -> tuple[float, float]:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt_input != pad_idx).unsqueeze(1).unsqueeze(2) & \
                   torch.tril(torch.ones(tgt_input.size(1), tgt_input.size(1), device=device)).bool()

        output = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(output, tgt_output)

        n_tokens = (tgt_output != pad_idx).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens, 0.0


def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    pad_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """Greedy decoding for inference."""
    model.eval()

    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).long().to(device)

    for _ in range(max_len - 1):
        tgt_mask = torch.tril(torch.ones(ys.size(1), ys.size(1), device=device)).bool()
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

        out = model.decode(ys, memory, tgt_mask, src_mask)
        prob = model.generator(out[:, -1, :])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word).long().to(device)], dim=1)

        if next_word == end_symbol:
            break

    return ys
