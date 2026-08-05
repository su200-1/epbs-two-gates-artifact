"""Recurrent (LSTM) actor-critic for PPO.

Structure mirrors BunnyFinder's `rl/model/actor_critic.py` (encoder -> LSTM ->
actor head + critic head + init_hidden) — the one sound piece of that repo's RL
code. Sized modestly for the small Tier 1 problem (6-dim obs, 3 actions,
<=8-step episodes); the LSTM is kept as scaffolding for Tier 2 (multi-slot).

Phase L-α (2026-05-25) extension — multi-head factorised action support.
The constructor accepts either:
  * ``n_actions: int`` — legacy single-head Categorical (Tier 1, exp11)
  * ``head_sizes: Sequence[int]`` — factorised action space, one Linear head
    per dim (Phase L-α 5-head: [3, 2, 3, 3, 2])

Forward always returns a **list of per-head logits**. Single-head mode
returns a 1-element list so callers can handle both uniformly. The
existing single-head PPO loop remains bit-identical because the legacy
shim wraps/unwraps the 1-element list. See `epbs/rl/ppo.py` for the
masked + autoregressive sampling logic that consumes multi-head output.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

Hidden = tuple[torch.Tensor, torch.Tensor]


class RecurrentActorCritic(nn.Module):
    def __init__(
        self, obs_dim: int,
        n_actions: int | None = None,
        *,
        head_sizes: Sequence[int] | None = None,
        hidden: int = 64, lstm_layers: int = 1,
    ):
        super().__init__()
        self.hidden = hidden
        self.lstm_layers = lstm_layers
        if head_sizes is None:
            if n_actions is None:
                raise ValueError("must specify either n_actions or head_sizes")
            head_sizes = (int(n_actions),)
        self.head_sizes: tuple[int, ...] = tuple(int(n) for n in head_sizes)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
        )
        self.lstm = nn.LSTM(hidden, hidden, num_layers=lstm_layers,
                            batch_first=True)
        # 5 heads in Phase L-α; 1 head in legacy mode. nn.ModuleList for
        # transparent parameter registration.
        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden, n) for n in self.head_sizes
        ])
        self.critic = nn.Linear(hidden, 1)

    @property
    def n_heads(self) -> int:
        return len(self.head_sizes)

    @property
    def total_head_dims(self) -> int:
        """Sum of per-head action dims — buffer mask width in PPO."""
        return sum(self.head_sizes)

    def forward(self, obs_seq: torch.Tensor, hidden_state: Hidden):
        """obs_seq: (batch, seq_len, obs_dim).

        Returns:
          * head_logits: list of per-head tensors each shape (batch, seq, n_h)
          * value: state values (batch, seq)
          * hidden_state: updated LSTM hidden state
        """
        x = self.encoder(obs_seq)
        x, hidden_state = self.lstm(x, hidden_state)
        head_logits = [head(x) for head in self.actor_heads]
        value = self.critic(x).squeeze(-1)
        return head_logits, value, hidden_state

    def init_hidden(self, batch_size: int = 1,
                    device: torch.device | str = "cpu") -> Hidden:
        shape = (self.lstm_layers, batch_size, self.hidden)
        return (torch.zeros(shape, device=device),
                torch.zeros(shape, device=device))
