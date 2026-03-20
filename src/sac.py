import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.utils.data_preprocessor import (
    format_flat_state,
    format_action,
    format_reward,
    format_terminal
)


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        policy,
        q1,
        q2,
        replay_buffer,
        horizon: int = 1,
        optimizer_str: str = "AdamW",
        policy_lr: float = 3e-4,
        q_lr: float = 1e-3,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        action_dim: int = 2,
        seed: int = 42,
        device="cpu",
    ):
        super(SoftActorCritic, self).__init__()

        self.horizon    = horizon
        self.batch_size = batch_size
        self.gamma      = gamma
        self.tau        = tau
        self.device     = device

        self.policy = policy.to(device)
        self.q1     = q1.to(device)
        self.q2     = q2.to(device)

        self.q1_target = copy.deepcopy(q1).to(device)
        self.q2_target = copy.deepcopy(q2).to(device)
        for p in self.q1_target.parameters(): p.requires_grad = False
        for p in self.q2_target.parameters(): p.requires_grad = False

        optimizer = getattr(optim, optimizer_str)
        self.policy_opt = optimizer(self.policy.parameters(), lr=policy_lr)
        self.q1_opt     = optimizer(self.q1.parameters(),     lr=q_lr)
        self.q2_opt     = optimizer(self.q2.parameters(),     lr=q_lr)

        # --- Adaptive entropy (auto-tune alpha) ---
        # Target entropy = -action_dim (standard heuristic from SAC paper)
        self.target_entropy = -float(action_dim)
        self.log_alpha      = torch.tensor(
            [float(alpha)], dtype=torch.float32, device=device
        ).log().requires_grad_(True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-4)

        self.replay_buffer = replay_buffer

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def soft_update(self):
        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.mul_(1 - self.tau); pt.data.add_(self.tau * p.data)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
            pt.data.mul_(1 - self.tau); pt.data.add_(self.tau * p.data)

    def update(self):
        self.policy.train()
        self.q1.train()
        self.q2.train()

        s, a, r, s_next, done = self.replay_buffer.sample(self.batch_size, self.horizon)

        s      = format_flat_state(s,      self.device)   # [B, H, state_dim]
        s_next = format_flat_state(s_next, self.device)   # [B, H, state_dim]
        r      = format_reward(r,    self.device).unsqueeze(-1).unsqueeze(-1)   # [B,1,1]
        done   = format_terminal(done, self.device).unsqueeze(-1).unsqueeze(-1) # [B,1,1]
        a      = format_action(a,    self.device).unsqueeze(1)                  # [B,1,A]

        # --- Target Q ---
        with torch.no_grad():
            next_a, next_log_pi = self.policy.sample(s_next, self.horizon)
            tq1 = self.q1_target(s_next, next_a)
            tq2 = self.q2_target(s_next, next_a)
            target_v = torch.min(tq1, tq2) - self.alpha * next_log_pi
            q_target = r + (1.0 - done) * self.gamma * target_v

        # --- Q update with gradient clipping ---
        q1_loss = F.mse_loss(self.q1(s, a), q_target)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=1.0)
        self.q1_opt.step()

        q2_loss = F.mse_loss(self.q2(s, a), q_target)
        self.q2_opt.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=1.0)
        self.q2_opt.step()

        # --- Policy update with gradient clipping ---
        curr_a, log_pi = self.policy(s, self.horizon)
        min_q = torch.min(self.q1(s, curr_a), self.q2(s, curr_a))
        policy_loss = (self.alpha.detach() * log_pi - min_q).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_opt.step()

        # --- Alpha update (adaptive entropy) ---
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.soft_update()