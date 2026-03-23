import re
import glob
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


LOGS_DIR = Path("logs")
OUT_DIR = Path("logs") / "training_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_run_number(run_path: Path):
    m = re.match(r"run_(\d+)", run_path.name)
    return int(m.group(1)) if m else None


def find_all_runs(logs_dir: Path):
    run_dirs = [p for p in logs_dir.glob("run_*") if p.is_dir()]
    run_dirs = [p for p in run_dirs if extract_run_number(p) is not None]
    run_dirs.sort(key=lambda p: extract_run_number(p))
    if not run_dirs:
        raise FileNotFoundError(f"No run_* folders found in {logs_dir.resolve()}")
    return run_dirs


def load_run_monitor_csvs(run_dir: Path) -> pd.DataFrame:
    csv_files = sorted(run_dir.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()

    dfs = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, skiprows=1)
            if {"r", "l", "t"}.issubset(df.columns):
                df["source_file"] = csv_path.name
                dfs.append(df)
        except Exception as e:
            print(f"Skipping {csv_path}: {e}")

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("t").reset_index(drop=True)
    out["episode_idx"] = range(len(out))
    out["return_ma_10"] = out["r"].rolling(window=10, min_periods=1).mean()
    out["length_ma_10"] = out["l"].rolling(window=10, min_periods=1).mean()
    return out


def build_epoch_summary(run_dirs):
    rows = []
    for run_dir in run_dirs:
        df = load_run_monitor_csvs(run_dir)
        run_num = extract_run_number(run_dir)
        if df.empty:
            rows.append({
                "run_name": run_dir.name,
                "epoch": run_num,
                "num_episodes": 0,
                "avg_return": float("nan"),
                "best_return": float("nan"),
                "worst_return": float("nan"),
                "avg_length": float("nan"),
                "max_length": float("nan"),
            })
            continue

        rows.append({
            "run_name": run_dir.name,
            "epoch": run_num,
            "num_episodes": len(df),
            "avg_return": df["r"].mean(),
            "best_return": df["r"].max(),
            "worst_return": df["r"].min(),
            "avg_length": df["l"].mean(),
            "max_length": df["l"].max(),
        })

    summary = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    summary["avg_return_ma_5"] = summary["avg_return"].rolling(window=5, min_periods=1).mean()
    summary["avg_length_ma_5"] = summary["avg_length"].rolling(window=5, min_periods=1).mean()
    return summary


def plot_latest_run(latest_run: Path):
    df = load_run_monitor_csvs(latest_run)
    if df.empty:
        print(f"No usable monitor CSVs found in {latest_run}")
        return

    print(f"\nLatest run: {latest_run.name}")
    print(df[["episode_idx", "r", "l", "t", "source_file"]].tail(10).to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.plot(df["episode_idx"], df["r"], label="Episode return")
    plt.plot(df["episode_idx"], df["return_ma_10"], label="Return MA(10)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"Episode Return - {latest_run.name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{latest_run.name}_episode_return.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df["episode_idx"], df["l"], label="Episode length")
    plt.plot(df["episode_idx"], df["length_ma_10"], label="Length MA(10)")
    plt.xlabel("Episode")
    plt.ylabel("Episode length")
    plt.title(f"Episode Length - {latest_run.name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{latest_run.name}_episode_length.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(df["r"], bins=30)
    plt.xlabel("Episode return")
    plt.ylabel("Count")
    plt.title(f"Return Distribution - {latest_run.name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{latest_run.name}_return_hist.png", dpi=200)
    plt.show()


def plot_all_epochs(summary: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.plot(summary["epoch"], summary["avg_return"], marker="o", label="Avg return")
    plt.plot(summary["epoch"], summary["avg_return_ma_5"], marker="o", label="Avg return MA(5)")
    plt.xlabel("Epoch")
    plt.ylabel("Average return")
    plt.title("Average Return Across All Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "all_epochs_avg_return.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(summary["epoch"], summary["best_return"], marker="o", label="Best episode return")
    plt.plot(summary["epoch"], summary["worst_return"], marker="o", label="Worst episode return")
    plt.xlabel("Epoch")
    plt.ylabel("Episode return")
    plt.title("Best / Worst Return Across All Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "all_epochs_best_worst_return.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(summary["epoch"], summary["avg_length"], marker="o", label="Avg episode length")
    plt.plot(summary["epoch"], summary["avg_length_ma_5"], marker="o", label="Avg length MA(5)")
    plt.xlabel("Epoch")
    plt.ylabel("Average episode length")
    plt.title("Average Episode Length Across All Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "all_epochs_avg_length.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(summary["epoch"], summary["num_episodes"])
    plt.xlabel("Epoch")
    plt.ylabel("Number of episodes logged")
    plt.title("Episodes Collected Per Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "all_epochs_num_episodes.png", dpi=200)
    plt.show()


def main():
    run_dirs = find_all_runs(LOGS_DIR)
    latest_run = run_dirs[-1]

    summary = build_epoch_summary(run_dirs)
    summary.to_csv(OUT_DIR / "epoch_summary.csv", index=False)

    print("\n===== Epoch Summary =====")
    print(summary.to_string(index=False))

    plot_latest_run(latest_run)
    plot_all_epochs(summary)

    print(f"\nSaved outputs to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()