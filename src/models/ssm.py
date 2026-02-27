import torch
import torch.nn as nn
import torch.nn.functional as F

class StateSpaceModel(nn.Module):
    """
    State-Space Model (SSM) Layer.

    This implements the discrete-time state-space system:

        x'_t = A * x_t + B * u_t      # State update equation
        y_t = C * x_t + D * u_t       # Output equation

    Where:
        - s_t ∈ ℝ^d_state is the latent state at time step t.
        - u_t ∈ ℝ^d_model is the input at time step t.
        - y_t ∈ ℝ^d_model is the output at time step t.
        - A ∈ ℝ^d_state (diagonal) controls the memory dynamics.
        - B ∈ ℝ^(d_state × d_model) maps input to state.
        - C ∈ ℝ^(d_model × d_state) maps state to output.

    Args:
        d_model (int): Model hidden dimension.
        d_state (int): Number of state variables.
    """
    def __init__(
        self, 
        d_model: int, 
        d_state: int,
    ):
        super(StateSpaceModel, self).__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A is a diagonal matrix (stored as a vector)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()))

        # Selection Mechanism (The 'Selective' in Selective SSM)
        # delta, B, and C are functions of the input x
        self.dt_rank = max(1, d_model // 16)
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # D Skip connection (Residual)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, u: torch.Tensor):
        """
        u: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = u.shape
        
        # Project entire sequence at once
        # x_proj_result: [batch, seq_len, dt_rank + 2*d_state]
        x_proj_result = self.x_proj(u)
        dt_raw, B, C = torch.split(x_proj_result, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Compute Delta and Discretize A
        delta = F.softplus(self.dt_proj(dt_raw)) # [batch, seq_len, d_model]
        A = -torch.exp(self.A_log)               # [d_state]
        
        # dA = exp(delta * A) -> [batch, seq_len, d_model, d_state]
        dA = torch.exp(torch.einsum('bsd,n->bsdn', delta, A))
        
        # dB = delta * B -> [batch, seq_len, d_model, d_state]
        dB = torch.einsum('bsd,bsn->bsdn', delta, B)

        # Simplified Selective Scan
        states = []
        curr_state = torch.zeros(batch, d_model, self.d_state).to(u.device)
        
        for t in range(seq_len):
            # curr_state = dA_t * prev_state + dB_t * u_t
            curr_state = dA[:, t] * curr_state + dB[:, t] * u[:, t].unsqueeze(-1)
            
            # y_t = state_t @ C_t
            y_t = torch.einsum('bdn,bn->bd', curr_state, C[:, t])
            states.append(y_t)

        return torch.stack(states, dim=1) + u * self.D