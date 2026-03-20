import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence

def flatten_state(state, device=None) -> torch.Tensor:
    tensors = []
    for key, value in state.items():
        if isinstance(value, dict):
            tensors.append(flatten_state(value))
        else:
            tensors.append(torch.as_tensor(value, dtype=torch.float32).flatten())
    tensor_cat = torch.cat(tensors, dim=0)
    return tensor_cat.to(device) if device else tensor_cat

def format_flat_state(state, device=None) -> torch.Tensor:
    """
    Convert a batch of flat numpy state arrays (from the buffer) to a tensor.
    Each item in state is a (horizon, state_dim) float32 numpy array.
    Output: [Batch, Horizon, state_dim]
    """
    t_list = [torch.as_tensor(item, dtype=torch.float32) for item in state]
    out = pad_sequence(t_list, batch_first=True, padding_value=0.0)
    # If horizon=1, pad_sequence may drop the sequence dim — restore it
    if out.dim() == 2:
        out = out.unsqueeze(1)  # [B, state_dim] -> [B, 1, state_dim]
    return out.to(device) if device else out

def format_state(state, device=None) -> torch.Tensor:
    """Recursive wrapper to maintain input nesting in the output."""

    if isinstance(state, dict):
        return flatten_state(state, device)

    elif isinstance(state, (list, np.ndarray)):
        r_list = [format_state(item, device) for item in state]

        if r_list[0].dim() == 1:
            return torch.stack(r_list)

        r_out = pad_sequence(r_list, batch_first=True, padding_value=0.0)

        if r_out.dim() == 4 and r_out.shape[2] == 1:
            r_out = r_out.squeeze(2)

        return r_out
    
    if not isinstance(state, torch.Tensor):
        return torch.as_tensor(state, device=device)
        
    return state.to(device) if device else state

def format_reward(reward, device=None) -> torch.Tensor:
    r_list = [torch.as_tensor(item, dtype=torch.float32) for item in reward]
    r_out = pad_sequence(r_list, batch_first=True, padding_value=0.0).squeeze()
    if device:
        r_out = r_out.to(device)
    return r_out

def format_terminal(terminal, device=None) -> torch.Tensor:
    t_list = [torch.as_tensor(item, dtype=torch.float32) for item in terminal]
    t_out = pad_sequence(t_list, batch_first=True, padding_value=1.0).squeeze()
    if device:
        t_out = t_out.to(device)
    return t_out

def format_action(action, device=None) -> torch.Tensor:
    a_list = [torch.as_tensor(item, dtype=torch.float32) for item in action]
    a_out = pad_sequence(a_list, batch_first=True, padding_value=0.0).squeeze()
    if device:
        a_out = a_out.to(device)
    return a_out