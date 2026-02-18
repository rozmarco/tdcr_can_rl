import torch
import torch.nn as nn

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
        d_state: int=16,
    ):
        super(StateSpaceModel, self).__init__()

        self.d_model = d_model
        self.d_state = d_state

        # Learnable state-space parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))  # Input-to-state
        self.C = nn.Parameter(torch.randn(d_model, d_state))  # State-to-output
        self.D = nn.Parameter(torch.randn(d_model))

        # Fixed step size (delta)
        self.log_delta = nn.Parameter(torch.tensor([0.0]))

    def forward(self, state: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SSM layer.
        """
        # Discretization (Zero-Order Hold)
        delta = torch.exp(self.log_delta)
        A_bar = torch.exp(-delta * self.A)
        B_bar = delta * self.B

        # State-space recurrence: s_t = A * s_t + B * u_t
        state = A_bar @ state + B_bar @ u_t.mT

        # Output transformation: y_t = C * s_t + D * u_t
        y = self.C @ state + self.D @ u_t.mT

        return y, state