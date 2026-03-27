import os
import glob
import yaml

import ray
import torch
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

from src.environment.envrunner import EnvRunner, ParallelEnvRunner
from src.models.policy_network import LatentPolicyPlanner
from src.models.q_network import QNetwork
from src.buffers.buffer import ReplayBuffer
from src.sac import SoftActorCritic


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_environment(env_cls, scene_path, workspace_npz, loc,
                    H_all_ref, goal_to_table_idx_ref, lookup_table_npz):  # ← add
    state_dict = {k: v.cpu() for k, v in policy_network.state_dict().items()}
    workers = [
        env_cls.remote(i, True, scene_path, workspace_npz, state_dict, config, loc,
                        H_all_ref=H_all_ref,
                        goal_to_table_idx_ref=goal_to_table_idx_ref,
                        lookup_table_npz=str(lookup_table_npz))           # ← add
        for i in range(config['env']['num_workers'])
    ]
    return ray.get([w.run_session_remote.remote() for w in workers])


def get_new_npz(rollout_dir: Path, already_loaded: set) -> list:
    """Return rollout npz files not yet loaded into the buffer."""
    npz_files = sorted(rollout_dir.rglob("*.npz"), key=lambda x: x.stat().st_mtime)
    return [str(p) for p in npz_files if str(p) not in already_loaded]


def get_avg_return(log_dir: Path, run_name: str) -> float:
    run_path = log_dir / run_name
    if not run_path.exists():
        return float('-inf')
    rewards = []
    for csv in glob.glob(str(run_path / "*.csv")):
        try:
            df = pd.read_csv(csv, skiprows=1)
            if "r" in df.columns and len(df) > 0:
                rewards.append(df["r"].mean())
        except Exception:
            continue
    return float(np.mean(rewards)) if rewards else float('-inf')


def save_checkpoint(label, models):
    os.makedirs("checkpoints", exist_ok=True)
    for name, model in models.items():
        torch.save(model.state_dict(), f"checkpoints/{name}_{label}.pth")


def save_best_checkpoint(models, avg_return: float):
    os.makedirs("checkpoints", exist_ok=True)
    for name, model in models.items():
        torch.save(model.state_dict(), f"checkpoints/{name}_best.pth")
    tqdm.write(f"\033[92mNew best model saved — avg return: {avg_return:.2f}\033[0m")


def load_checkpoint(network, checkpoint_path):
    if checkpoint_path is None:
        print(f"\033[93mNo weights requested for {type(network).__name__}.\033[0m")
        return
    if str(checkpoint_path).strip().lower() in ["none", ""]:
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

    with open(PROJECT_ROOT_ / "config" / "train.yaml", "r") as f:
        config = yaml.safe_load(f)

    scene_path = Path(config["scene"])
    if not scene_path.is_absolute():
        scene_path = PROJECT_ROOT_ / scene_path

    workspace_npz = Path(config["workspace_npz"])
    if not workspace_npz.is_absolute():
        workspace_npz = PROJECT_ROOT_ / workspace_npz

    rollout_dir = Path(config["env"]["rollout_data_dir"])
    if not rollout_dir.is_absolute():
        rollout_dir = PROJECT_ROOT_ / rollout_dir
    rollout_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = PROJECT_ROOT_ / "logs"

    set_seed(config["seed"])

    policy_network = LatentPolicyPlanner(
        r_dim=config["agent"]["r_dim"],
        action_dim=config["agent"]["action_dim"],
        **config['model']["policy"]
    )
    q_network1 = QNetwork(
        r_dim=config["agent"]["r_dim"],
        action_dim=config["agent"]["action_dim"],
        **config["model"]["q_network"]
    )
    q_network2 = QNetwork(
        r_dim=config["agent"]["r_dim"],
        action_dim=config["agent"]["action_dim"],
        **config["model"]["q_network"]
    )

    load_checkpoint(policy_network, config["model"]["policy"]["checkpoint"])
    load_checkpoint(q_network1,     config["model"]["q_network"]["checkpoint_q1"])
    load_checkpoint(q_network2,     config["model"]["q_network"]["checkpoint_q2"])

    for net in [policy_network, q_network1, q_network2]:
        n = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"\033[92mInitialized {type(net).__name__} with {n} parameters.\033[0m")

    buffer = ReplayBuffer(max_size=config["buffer"]["max_size"], seed=config["seed"])

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
        action_dim=config["agent"]["action_dim"],
        seed=config["seed"],
        device=config["device"]
    )

    models_to_save  = {"policy": policy_network, "q1": q_network1, "q2": q_network2}
    total_epochs    = config['epochs']
    updates_per_epoch = config['agent']['updates_per_epoch']
    checkpoint_freq   = config['agent']['checkpoint']
    best_return       = float('-inf')
    loaded_npz        = set()
    total_updates     = 0

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

    ray.init(ignore_reinit_error=True)

    # ── Put the heavy arrays in Ray shared memory ONCE ───────────────────
    # All workers will read from this single copy instead of each loading
    # their own 221MB HeuristicTable.
    lookup_npz_path = Path(config["env"]["lookup_table_npz"])
    if not lookup_npz_path.is_absolute():
        lookup_npz_path = PROJECT_ROOT_ / lookup_npz_path
    _d = np.load(str(lookup_npz_path), allow_pickle=True)
    H_all_ref             = ray.put(_d["H_all"])
    goal_to_table_idx_ref = ray.put(_d["goal_to_table_idx"])
    del _d  # free local copy immediately

    for epoch in tqdm(range(total_epochs), desc="Epochs", leave=True):
        run_name = f"run_{epoch}"

        # --- Collect rollouts ---
        if config['env']['run_env']:
            policy_network.eval()
            with torch.no_grad():
                results = run_environment(
                    ParallelEnvRunner,
                    scene_path=str(scene_path),
                    workspace_npz=str(workspace_npz),
                    loc=run_name,
                    H_all_ref=H_all_ref,
                    goal_to_table_idx_ref=goal_to_table_idx_ref,
                    lookup_table_npz=lookup_npz_path,             # ← add
                )
            tqdm.write(f"\033[92mCompleted {len(results)} parallel sessions.\033[0m")

            avg_return = get_avg_return(logs_dir, run_name)
            tqdm.write(f"\033[92mEpoch {epoch} avg return: {avg_return:.2f}\033[0m")
            if avg_return > best_return:
                best_return = avg_return
                save_best_checkpoint(models_to_save, avg_return)

        # --- Load only new rollout files (buffer accumulates) ---
        new_files = get_new_npz(rollout_dir, loaded_npz)
        if new_files:
            buffer.load(new_files)
            loaded_npz.update(new_files)
            tqdm.write(f"\033[92mLoaded {len(new_files)} new files — buffer size: {len(buffer)}\033[0m")

        # --- Fixed number of SAC updates ---
        if buffer.can_sample(config["agent"]["horizon"]):
            for i in tqdm(range(updates_per_epoch), desc="SAC updates", leave=False):
                sac.update()
                total_updates += 1
                if total_updates % checkpoint_freq == 0:
                    save_checkpoint(total_updates, models_to_save)
            tqdm.write(f"\033[92mFinished {updates_per_epoch} SAC updates (total: {total_updates}).\033[0m")
        else:
            tqdm.write(f"\033[93mBuffer too small to sample — skipping SAC updates.\033[0m")

    save_checkpoint("final", models_to_save)
    tqdm.write(f"\033[92mTraining complete. Best return: {best_return:.2f}\033[0m")
    tqdm.write(f"\033[92mBest weights: checkpoints/*_best.pth\033[0m")