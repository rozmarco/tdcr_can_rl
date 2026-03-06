import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.utils.data_preprocessor import (
    format_state, 
    format_action,
    format_reward, 
    format_terminal
)

class SoftActorCritic(nn.Module):
    def __init__(
        self, 
        policy: nn.Module,
        q1: nn.Module,
        q2: nn.Module,
        replay_buffer,
        horizon: int = 10,
        optimizer_str: str = "AdamW",
        policy_lr: float = 3e-4, 
        q_lr: float = 1e-3,
        batch_size: int = 64,
        gamma: float = 0.99, 
        tau: float = 0.005, 
        alpha: float = 0.2,
        seed: int = 42,
        device: torch.device = "cpu",
    ):
        super(SoftActorCritic, self).__init__()

        self.horizon = horizon
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.seed = seed

        self.policy = policy.to(device)
        self.q1 = q1.to(device)
        self.q2 = q2.to(device)

        self.q1_target = copy.deepcopy(q1)
        self.q2_target = copy.deepcopy(q2)
        for p in self.q1_target.parameters(): p.requires_grad = False
        for p in self.q2_target.parameters(): p.requires_grad = False

        optimizer = getattr(optim, optimizer_str)
        self.policy_opt = optimizer(self.policy.parameters(), lr=policy_lr)
        self.q1_opt = optimizer(self.q1.parameters(), lr=q_lr)
        self.q2_opt = optimizer(self.q2.parameters(), lr=q_lr)

        self.replay_buffer = replay_buffer

    @torch.no_grad()
    def soft_update(self):
        """
        Gradually shifts target weights toward live weights.
        target = tau * live + (1 - tau) * target
        """
        # Update Target 1
        for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
            p_targ.data.mul_(1 - self.tau)
            p_targ.data.add_(self.tau * p.data)
        # Update Target 2
        for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
            p_targ.data.mul_(1 - self.tau)
            p_targ.data.add_(self.tau * p.data)

    def update(self):
        self.policy.train()
        self.q1.train()
        self.q2.train()
    
        s, a, r, s_next, done = self.replay_buffer.sample(self.batch_size, self.horizon)

        s = format_state(s, self.device)
        s_next = format_state(s_next, self.device)
        r = format_reward(r, self.device)
        done = format_terminal(done, self.device)
        a = format_action(a, self.device)

        # --- Compute the Target Q-value ---
        with torch.no_grad():
            next_action, next_log_pi = self.policy.rollout(s, self.horizon)
            
            # Use TARGET Q-networks for stability
            target_q1_next = self.q1_target(s_next, next_action)
            target_q2_next = self.q2_target(s_next, next_action)
            
            # The "Soft" State Value: min(Q) - alpha * entropy
            min_target_q = torch.min(target_q1_next, target_q2_next)
            target_v = min_target_q - self.alpha * next_log_pi
            
            # The Bellman Equation
            q_target = r + (1.0 - done) * self.gamma * target_v

        # --- Update Live Q-Networks ---
        # Use Q-networks to predict the q_target we just calculated
        q1_loss = F.mse_loss(self.q1(s, a), q_target)
        q2_loss = F.mse_loss(self.q2(s, a), q_target)
        
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()
        
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()
        
        # --- Update Policy (Reverse KL) ---
        # Use Q-networks here to tell the policy which way to move
        curr_a, log_pi = self.policy.rollout(s, self.horizon)
        q1_pi = self.q1(s, curr_a)
        q2_pi = self.q2(s, curr_a)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        # Minimize (alpha * log_pi - Q) is the same as maximizing (Q - alpha * log_pi)
        # KL(pi || exp(Q/alpha)) = E[αlogπ(a∣s) - Q(s,a)]
        policy_loss = (self.alpha * log_pi - min_q_pi).mean()
        
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # --- Soft Update Target Q-Networks ---
        # Instead of updating V, update the Q-targets
        self.soft_update()
        self.soft_update()