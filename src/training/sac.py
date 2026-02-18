import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SoftActorCritic(nn.Module):
    def __init__(
        self, 
        policy,
        q1,
        q2,
        replay_buffer,
        lr=3e-4, 
        batch_size=64,
        gamma=0.99, 
        tau=0.005, 
        alpha=0.2
    ):
        super(SoftActorCritic, self).__init__()
        
        self.policy = policy
        self.q1 = q1
        self.q2 = q2
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.q1_target = copy.deepcopy(q1)
        self.q2_target = copy.deepcopy(q2)
        for p in self.q1_target.parameters(): p.requires_grad = False
        for p in self.q2_target.parameters(): p.requires_grad = False

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)

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
        s, a, r, s_next, done = self.replay_buffer.sample(self.batch_size)
        
        # --- Compute the Target Q-value ---
        with torch.no_grad():
            next_action, next_log_pi = self.policy.sample(s_next)
            
            # Use TARGET Q-networks for stability
            target_q1_next = self.q1_target(s_next, next_action)
            target_q2_next = self.q2_target(s_next, next_action)
            
            # The "Soft" State Value: min(Q) - alpha * entropy
            min_target_q = torch.min(target_q1_next, target_q2_next)
            target_v = min_target_q - self.alpha * next_log_pi
            
            # The Bellman Equation
            q_target = r + (1 - done) * self.gamma * target_v

        # --- Update Live Q-Networks ---
        # We want our live Q-networks to predict the q_target we just calculated
        q1_loss = F.mse_loss(self.q1(s, a), q_target)
        q2_loss = F.mse_loss(self.q2(s, a), q_target)
        
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()
        
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()
        
        # --- Update Policy (Reverse KL) ---
        # We use the live Q-networks here to tell the policy which way to move
        curr_a, log_pi = self.policy.sample(s)
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
        # Instead of updating V, we update the Q-targets
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)
