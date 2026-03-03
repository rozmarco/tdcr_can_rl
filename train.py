import yaml
import ray
import torch

from tqdm import tqdm
from pathlib import Path
from itertools import count

from src.environment.envrunner import EnvRunner, ParallelEnvRunner
from src.models.policy_network import LatentDiffusionPolicyPlanner
from src.models.q_network import QNetwork
from src.buffers.buffer import ReplayBuffer
from src.sac import SoftActorCritic

from tdcr_sim_mujoco.src.utils.config_loader import PROJECT_ROOT


def run_environment(env):
    workers = [env.remote(True, scene_path, policy_network, config) 
                for _ in range(config['env']['num_workers'])]
    future_results = [w.run_session_remote.remote() for w in workers]
    results = ray.get(future_results)
    return results

def get_sorted_npz():
    npz_files = list(Path("data").rglob("*.npz"))
    npz_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    npz_files = [str(p) for p in npz_files]
    return npz_files

def save_checkpoint(it, models):
    # Overwrite the previous checkpoints
    for name, model in models.items():
        torch.save(model.state_dict(), f"checkpoints/{name}_{it}.pth")


if __name__ == '__main__':
    PROJECT_ROOT_ = Path(__file__).parent.resolve()

    # ----- YAML CONFIGURATION -----
    config_file = Path(PROJECT_ROOT_ / "config" / "train.yaml")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # ----- INITIALIZATION -----
    torch.manual_seed(config["seed"])

    r_dim = config["agent"]["r_dim"]
    action_dim = config["agent"]["action_dim"]

    policy_network = LatentDiffusionPolicyPlanner(
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

    # ----- TRAINING -----
    ray.init(ignore_reinit_error=True)

    for epoch in tqdm(range(config['epochs']), desc="Training Epochs"):
        
        # Run environment
        if config['env']['run_env']:
            with torch.no_grad():
                policy_network.eval()
                results = run_environment(ParallelEnvRunner)
                tqdm.write(f"\033[92mCompleted {len(results)} parallel sessions.\033[0m")

        # Load buffer
        buffer.clear()
        npz_files = get_sorted_npz()
        buffer.load(npz_files)
        tqdm.write(f"\033[92mLoaded {len(buffer)} samples.\033[0m")

        # Training loop
        for iteration in count():
            if not buffer.can_sample(batch_size):
                break

            sac.update()

            if iteration % checkpoint == 0:
                save_checkpoint(iteration, models_to_save)

    save_checkpoint(iteration, models_to_save)