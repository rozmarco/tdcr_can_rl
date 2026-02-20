import torch
import torch_geometric

from src.models.gnn import generate_edges

def format_state(state, device=None):
    # Format robot state
    robot_data = state['robot']
    goal_data = state['goal']
    ten = torch.as_tensor(robot_data['ten_length'], dtype=torch.float32).reshape(-1)
    qfrc = torch.as_tensor(robot_data['qfrc'], dtype=torch.float32).reshape(-1)
    radii = torch.as_tensor(robot_data['link_radii'], dtype=torch.float32).reshape(-1)
    # TODO: Add link_pose or curvature here
    goal_pos = torch.as_tensor(goal_data['position'], dtype=torch.float32).reshape(-1)
    goal_ori = torch.as_tensor(goal_data['orientation'], dtype=torch.float32).reshape(-1)
    r_state = torch.cat([ten, qfrc, radii, goal_pos, goal_ori], dim=0).unsqueeze(0)

    # Format robot + obstacle state
    obs_data = state['obstacles']
    obs_pos = torch.as_tensor(obs_data['rel_positions'], dtype=torch.float32)
    obs_rad = torch.as_tensor(obs_data['radii'], dtype=torch.float32)
    obstacles_tensor = torch.cat([obs_pos, obs_rad], dim=-1)

    r_temp = torch.rand((1, 3 + 1)) # TODO: Replace with proper position and orientation
    obstacles_tensor = torch.cat([obstacles_tensor, r_temp], dim=0)

    n_obstacles = obstacles_tensor.shape[0]
    o_graph = torch_geometric.data.Data(x=obstacles_tensor, edge_index=generate_edges(n_obstacles))
    
    # Format initial ssm state
    ssm_state = torch.randn((64, 1))

    if device is not None:
        r_state = r_state.to(device)
        o_graph = o_graph.to(device)
        ssm_state = ssm_state.to(device)

    return r_state, o_graph, ssm_state