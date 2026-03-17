import os
import yaml

import ray
import torch
import random
import numpy as np

from tqdm import tqdm
from pathlib import Path
from itertools import count

from src.environment.envrunner import EnvRunner, ParallelEnvRunner
from src.models.policy_network import LatentPolicyPlanner
from src.models.q_network import QNetwork
from src.buffers.buffer import ReplayBuffer
from src.sac import SoftActorCritic

from tdcr_sim_mujoco.src.utils.config_loader import PROJECT_ROOT


def set_seed(seed: int=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_environment(env, loc):
    workers = [env.remote(i, True, scene_path, policy_network, config, loc) 
                for i in range(config['env']['num_workers'])]
    future_results = [w.run_session_remote.remote() for w in workers]
    results = ray.get(future_results)
    return results

def get_sorted_npz():
    npz_files = list(Path("data").rglob("*.npz"))
    npz_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    npz_files = [str(p) for p in npz_files]
    return npz_files

def save_checkpoint(it, models):
    os.makedirs("checkpoints", exist_ok=True)

    for name, model in models.items():
        path = f"checkpoints/{name}_{it}.pth"
        if not os.path.exists(path):
            torch.save(model.state_dict(), path)

def load_checkpoint(network, checkpoint_path):
    if checkpoint_path.strip().lower() in ["none", ""]:
        print(f"\033[93mNo weights requested for {type(network).__name__}.\033[0m")
        return
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT_ / checkpoint_path

    if not checkpoint_path.exists():
        print(f"\033[93mCould not find checkpoint: {checkpoint_path}.\033[0m")
        return

    network.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)


if __name__ == '__main__':
    PROJECT_ROOT_ = Path(__file__).parent.resolve()


    # ----- YAML CONFIGURATION -----
    config_file = Path(PROJECT_ROOT_ / "config" / "train.yaml")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)


    # ----- INITIALIZATION -----
    set_seed(config["seed"])

    r_dim = config["agent"]["r_dim"]
    action_dim = config["agent"]["action_dim"]

    policy_network = LatentPolicyPlanner(
        r_dim=r_dim,
        action_dim=action_dim,
        **config['model']["policy"]
    )
    q_network1 = QNetwork(
        r_dim=r_dim,
        action_dim=action_dim,
        **config["model"]["q_network"]
    )
    q_network2 = QNetwork(
        r_dim=r_dim,
        action_dim=action_dim,
        **config["model"]["q_network"]
    )

    load_checkpoint(policy_network, config["model"]["policy"]["checkpoint"])
    load_checkpoint(q_network1, config["model"]["q_network"]["checkpoint_q1"])
    load_checkpoint(q_network2, config["model"]["q_network"]["checkpoint_q2"])

    for net in [policy_network, q_network1, q_network2]:
        n_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"\033[92mInitialized {type(net).__name__} with {n_param} parameters.\033[0m")

    buffer = ReplayBuffer(
        max_size=config["buffer"]["max_size"],
        seed=config["seed"]
    )

    sac = SoftActorCritic(
        policy=policy_network,
        q1=q_network1,
        q2=q_network2,
        replay_buffer=buffer,
        horizon=config["agent"]["horizon"],
        optimizer_str=config["agent"]["optimizer"]["_target_"],
        policy_lr=config["agent"]["policy_lr"],
        q_lr=config["agent"]["q_lr"],
        batch_size=config["agent"]["batch_size"],
        gamma=config["agent"]["gamma"],
        tau=config["agent"]["tau"],
        alpha=config["agent"]["alpha"],
        seed=config["seed"],
        device=config["device"]
    )

    scene_path = Path(config["scene"])
    if not scene_path.is_absolute():
        scene_path = PROJECT_ROOT / scene_path

    models_to_save = {"policy": policy_network, "q1": q_network1, "q2": q_network2}
    checkpoint = config['agent']['checkpoint']
    batch_size = config["agent"]["batch_size"]
    total_epochs = config['epochs']


    # ----- TRAINING -----
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    ray.init(ignore_reinit_error=True)

    for epoch in tqdm(range(total_epochs), desc="Epochs", leave=True):

        # Run Environment (Simulate)
        if config['env']['run_env']:
            policy_network.eval()
            with torch.no_grad():
                results = run_environment(ParallelEnvRunner, loc=f"run_{epoch}") # loc = 'logs/run_0' folder name
            tqdm.write(f"\033[92mCompleted {len(results)} parallel sessions.\033[0m")

        # Load Buffer
        buffer.clear()
        npz_files = get_sorted_npz()
        buffer.load(npz_files)
        tqdm.write(f"\033[92mLoaded {len(buffer)} samples.\033[0m")

        # Run Training
        with tqdm(desc="Training Iteration", leave=False) as pbar:
            for iteration in count(): # Iteration does not reset to 0
                if not buffer.can_sample(horizon=config["agent"]["horizon"]):
                    break

                sac.update()

                if iteration % checkpoint == 0:
                    save_checkpoint(iteration, models_to_save)

                pbar.update(1)
        tqdm.write(f"\033[92mFinished SAC training.\033[0m")

    save_checkpoint(f"{iteration}_final", models_to_save)