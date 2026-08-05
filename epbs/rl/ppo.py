"""Proximal Policy Optimization for the recurrent free-option agent.

A correct PPO implementation — rollout buffer, per-episode GAE, clipped
surrogate objective, value loss, entropy bonus, minibatch multi-epoch updates,
and proper per-episode LSTM hidden-state handling.

NOTE: BunnyFinder's `rl/ppo/ppo_agent.py` is *not* reused — it has no rollout
buffer, no GAE, updates every single step with batch size 1, and re-inits the
LSTM hidden state on every update (destroying the recurrence). This module is
written from scratch.

Phase L-α (2026-05-25) — multi-head + masked + autoregressive support.
The PPO class now handles factorised action spaces (5 heads in L-α: P,
B_bid, B_reveal, C, PTC), masked sampling per head, and an autoregressive
sampling order (B_reveal sampled AFTER B_bid because its mask depends on
B_bid's choice — whether the byz builder will actually win the auction
under the chosen bid magnitude).

Math correctness: under independent + conditional categorical heads,
joint policy = product of head categoricals, so joint log_prob = sum of
head log_probs. PPO's clip + GAE + value loss operate unchanged on the
summed log_probs (this is the standard PPO treatment of factorised
action spaces).

Backward compatibility: when the model has a single head, the PPO loop
reproduces the legacy single-Categorical behaviour bit-identically.
Mask buffers are populated with all-True when no mask_callback is set.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from epbs.rl.actor_critic import RecurrentActorCritic


# Sentinel used by PPO.act when no mask is provided (single-head /
# pre-L-α callers). The rollout buffer stores all-True masks so the
# update-time masked categorical reconstruction is a no-op.
_NEG_INF = -1e9


# --------------------------------------------------------------------------
# Rollout storage
# --------------------------------------------------------------------------
class RolloutBuffer:
    """Collects whole episodes. Episodes truly terminate (no bootstrap).

    Phase L-α: each per-step entry's ``action`` is a list of ints (one per
    head; length 1 in single-head mode). ``mask`` is a flat list of bools
    of length ``sum(head_sizes)`` (all True if the caller didn't pass a
    mask). The PPO update consumes these via the model's ``head_sizes``.
    """

    def __init__(self):
        self.episodes: list[dict] = []
        self._cur: dict | None = None

    def start_episode(self) -> None:
        self._cur = {"obs": [], "action": [], "logp": [], "reward": [],
                     "value": [], "mask": []}

    def add(self, obs, action, logp, reward, value, mask=None) -> None:
        assert self._cur is not None, "start_episode() first"
        self._cur["obs"].append(np.asarray(obs, dtype=np.float32))
        # Normalise action to a list[int] regardless of single-/multi-head.
        if isinstance(action, (list, tuple, np.ndarray)):
            a_list = [int(a) for a in action]
        else:
            a_list = [int(action)]
        self._cur["action"].append(a_list)
        self._cur["logp"].append(float(logp))
        self._cur["reward"].append(float(reward))
        self._cur["value"].append(float(value))
        # Mask: None → defer to update-time fill with all-True
        self._cur["mask"].append(None if mask is None else [bool(b) for b in mask])

    def end_episode(self) -> None:
        assert self._cur is not None
        self.episodes.append(self._cur)
        self._cur = None

    def clear(self) -> None:
        self.episodes = []
        self._cur = None

    def __len__(self) -> int:
        return len(self.episodes)


def compute_gae(rewards: list[float], values: list[float],
                gamma: float, lam: float) -> tuple[list[float], list[float]]:
    """GAE for one terminated episode. Returns (advantages, returns)."""
    n = len(rewards)
    adv = [0.0] * n
    gae = 0.0
    for t in reversed(range(n)):
        next_value = values[t + 1] if t + 1 < n else 0.0  # terminal => 0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    returns = [adv[t] + values[t] for t in range(n)]
    return adv, returns


MaskCallback = Callable[[int, list[int | None]], "list[bool] | np.ndarray"]


# --------------------------------------------------------------------------
# PPO
# --------------------------------------------------------------------------
class PPO:
    def __init__(self, model: RecurrentActorCritic, *,
                 lr: float = 3e-4, clip: float = 0.2,
                 gamma: float = 0.99, lam: float = 0.95,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 epochs: int = 6, minibatch_episodes: int = 64,
                 max_grad_norm: float = 0.5, max_seq_len: int = 8,
                 device: str = "cpu",
                 sample_order: list[int] | None = None):
        self.model = model.to(device)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip = clip
        self.gamma = gamma
        self.lam = lam
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.minibatch_episodes = minibatch_episodes
        self.max_grad_norm = max_grad_norm
        self.max_seq_len = max_seq_len
        self.device = device
        # Autoregressive sampling order. Defaults to ascending head index
        # (no autoregression). Phase L-α 5-head callers pass [0,1,3,4,2]
        # so B_reveal (index 2) is sampled LAST, after B_bid (index 1).
        # The order does NOT affect joint log_prob (product of conditionals)
        # — it just decides which head's mask sees which prior action.
        if sample_order is None:
            sample_order = list(range(model.n_heads))
        if sorted(sample_order) != list(range(model.n_heads)):
            raise ValueError(
                f"sample_order={sample_order} must be a permutation of "
                f"range({model.n_heads})")
        self.sample_order = list(sample_order)
        # Diagnostic: most recently sampled mask (flat over head dims).
        # Callers (e.g., CoalitionEnvL.step) read this to push into the
        # rollout buffer so update-time mask matches sampling-time mask.
        self._last_mask_flat: list[bool] = []

    # -- rollout-time acting ----------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, hidden, *,
            mask_callback: MaskCallback | None = None,
            deterministic: bool = False):
        """One environment step.

        Returns (action, logp, value, new_hidden):
          * action — int (single-head) or list[int] (multi-head)
          * logp — sum of per-head log_probs (joint log_prob)
          * value — critic estimate
          * new_hidden — updated LSTM state

        Side effect: ``self._last_mask_flat`` set to the flat list of bools
        (length ``sum(head_sizes)``) used during sampling. Callers feed
        this into ``RolloutBuffer.add(..., mask=ppo._last_mask_flat)`` to
        keep update-time masking consistent with sampling-time.

        ``mask_callback(head_idx, prev_actions)`` is queried in
        ``self.sample_order`` and receives the in-progress action list
        (None for not-yet-sampled heads). Must return a bool sequence of
        length head_sizes[head_idx]; entries set to False force probability
        to zero on that primitive.
        """
        x = torch.as_tensor(obs, dtype=torch.float32,
                            device=self.device).view(1, 1, -1)
        head_logits, value, hidden = self.model(x, hidden)
        head_sizes = self.model.head_sizes

        actions: list[int | None] = [None] * self.model.n_heads
        masks_per_head: list[list[bool]] = [[] for _ in range(self.model.n_heads)]
        total_logp = 0.0

        for h in self.sample_order:
            logits_h = head_logits[h][:, 0, :]  # (1, n_h)
            n_h = head_sizes[h]
            if mask_callback is not None:
                raw = mask_callback(h, actions)
                mask_h = [bool(b) for b in raw]
                if len(mask_h) != n_h:
                    raise ValueError(
                        f"mask_callback returned {len(mask_h)} bools for head "
                        f"{h} but head_sizes[{h}]={n_h}")
                if not any(mask_h):
                    raise ValueError(
                        f"mask_callback returned all-False for head {h}; "
                        "every head must have at least one legal primitive")
                mask_t = torch.tensor(mask_h, dtype=torch.bool,
                                      device=self.device).view(1, -1)
                logits_h = logits_h.masked_fill(~mask_t, _NEG_INF)
            else:
                mask_h = [True] * n_h
            masks_per_head[h] = mask_h
            dist_h = torch.distributions.Categorical(logits=logits_h)
            if deterministic:
                a_h_t = torch.argmax(dist_h.probs, dim=-1)
            else:
                a_h_t = dist_h.sample()
            actions[h] = int(a_h_t.item())
            total_logp += float(dist_h.log_prob(a_h_t).item())

        # Flatten masks in head-index order for buffer storage.
        flat_mask: list[bool] = []
        for h in range(self.model.n_heads):
            flat_mask.extend(masks_per_head[h])
        self._last_mask_flat = flat_mask

        if self.model.n_heads == 1:
            # Legacy single-head: return int action, NOT a 1-list, so
            # pre-L-α callers (exp11) keep working unchanged.
            return (actions[0], total_logp,
                    float(value[0, 0].item()), hidden)
        return (list(actions), total_logp,
                float(value[0, 0].item()), hidden)

    def fresh_hidden(self, batch_size: int = 1):
        return self.model.init_hidden(batch_size, self.device)

    # -- training update ---------------------------------------------------
    def update(self, buffer: RolloutBuffer) -> dict:
        eps = buffer.episodes
        n = len(eps)
        L, obs_dim = self.max_seq_len, len(eps[0]["obs"][0])
        n_heads = self.model.n_heads
        head_sizes = self.model.head_sizes
        total_dims = self.model.total_head_dims

        obs = torch.zeros(n, L, obs_dim, device=self.device)
        # Action storage: (n, L, n_heads) int64
        act = torch.zeros(n, L, n_heads, dtype=torch.long, device=self.device)
        # Mask storage: (n, L, total_dims) bool, default all True
        buf_mask = torch.ones(n, L, total_dims, dtype=torch.bool,
                              device=self.device)
        old_logp = torch.zeros(n, L, device=self.device)
        adv = torch.zeros(n, L, device=self.device)
        ret = torch.zeros(n, L, device=self.device)
        mask = torch.zeros(n, L, device=self.device)  # valid-step mask

        for i, ep in enumerate(eps):
            a, r = compute_gae(ep["reward"], ep["value"], self.gamma, self.lam)
            t = len(ep["reward"])
            obs[i, :t] = torch.as_tensor(np.array(ep["obs"]), device=self.device)
            ep_actions = np.asarray(ep["action"], dtype=np.int64)  # (t, n_heads)
            if ep_actions.ndim == 1:
                ep_actions = ep_actions[:, None]  # legacy single-head saved as ints
            act[i, :t] = torch.as_tensor(ep_actions, device=self.device)
            for ti, m_step in enumerate(ep["mask"]):
                if m_step is not None:
                    buf_mask[i, ti] = torch.as_tensor(m_step, dtype=torch.bool,
                                                     device=self.device)
            old_logp[i, :t] = torch.as_tensor(ep["logp"], device=self.device)
            adv[i, :t] = torch.as_tensor(a, dtype=torch.float32, device=self.device)
            ret[i, :t] = torch.as_tensor(r, dtype=torch.float32, device=self.device)
            mask[i, :t] = 1.0

        # normalise advantages over all valid steps
        valid = mask.bool()
        adv_mean = adv[valid].mean()
        adv_std = adv[valid].std().clamp_min(1e-8)
        adv = (adv - adv_mean) / adv_std

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_updates": 0}
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, self.minibatch_episodes):
                idx = perm[s:s + self.minibatch_episodes]
                mb = len(idx)
                h0 = self.model.init_hidden(mb, self.device)
                head_logits, values, _ = self.model(obs[idx], h0)
                # Per-head masked log_prob and entropy, summed.
                mb_mask = buf_mask[idx]  # (mb, L, total_dims)
                mb_act = act[idx]        # (mb, L, n_heads)
                offset = 0
                new_logp = torch.zeros(mb, L, device=self.device)
                entropy_sum = torch.zeros(mb, L, device=self.device)
                for h, logits_h in enumerate(head_logits):
                    n_h = head_sizes[h]
                    mask_h = mb_mask[:, :, offset:offset + n_h]
                    logits_h = logits_h.masked_fill(~mask_h, _NEG_INF)
                    dist_h = torch.distributions.Categorical(logits=logits_h)
                    new_logp = new_logp + dist_h.log_prob(mb_act[:, :, h])
                    entropy_sum = entropy_sum + dist_h.entropy()
                    offset += n_h

                m = mask[idx]
                ratio = torch.exp(new_logp - old_logp[idx])
                a_mb = adv[idx]
                unclipped = ratio * a_mb
                clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a_mb
                policy_loss = -_masked_mean(torch.min(unclipped, clipped), m)
                value_loss = _masked_mean((values - ret[idx]) ** 2, m)
                entropy = _masked_mean(entropy_sum, m)

                loss = (policy_loss + self.value_coef * value_loss
                        - self.entropy_coef * entropy)
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["n_updates"] += 1

        k = max(1, stats["n_updates"])
        return {"policy_loss": stats["policy_loss"] / k,
                "value_loss": stats["value_loss"] / k,
                "entropy": stats["entropy"] / k}


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)
