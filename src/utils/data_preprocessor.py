import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence

def flatten_state(state, device=None) -> torch.Tensor:
    tensors = []
    for key, value in state.items():
        if isinstance(value, dict):
            tensors.append(flatten_state(value)) # If it's a sub-dictionary, go deeper
        else:
            tensors.append(torch.as_tensor(value, dtype=torch.float32).flatten())
    tensor_cat = torch.cat(tensors, dim=0)
    return tensor_cat.to(device) if device else tensor_cat

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
    
    # Pad shorter sequences, though risky as 0.0 means something to reward
    r_out = pad_sequence(r_list, batch_first=True, padding_value=0.0).squeeze()

    if device:
        r_out = r_out.to(device)

    return r_out

def format_terminal(terminal, device=None) -> torch.Tensor:
    t_list = [torch.as_tensor(item, dtype=torch.float32) for item in terminal]

    # Pad shorter sequences with 1.0 to end sequence in bellman equation
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