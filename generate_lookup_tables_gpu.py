"""
generate_lookup_tables_gpu.py
==============================
GPU-accelerated backward-Dijkstra lookup table generator.
Implements the heuristic cost-to-go H(s, g) from:
  Rao et al., "Towards Contact-Aided Motion Planning for TDCRs", RA-L 2024.

Used for potential-based RL reward shaping:
  r_shaped = r_task + Φ(s_{t+1}) - Φ(s_t)
  Φ(s) = min_{θ-bins} H[ix, iy, :]   (paper Section III-D smoothing)

Strategy
--------
Phase 1  [GPU, runs ONCE]
    Evaluate every (ix, iy, it, kappa, s) arc on the GPU in one batched
    CuPy kernel. Result: boolean mask (Nx, Ny, Ntheta, n_kappa, n_s).
    Saved to arc_cache.npz — never recomputed unless deleted.

Phase 2  [CPU, runs per UNIQUE goal cell]
    Deduplicate all ~8000 raw goals to unique (ix, iy, θ-bin) keys.
    Run backward Dijkstra once per unique key.
    Store a goal_to_table_idx mapping so every original goal index
    is addressable in RL via:  H_all[goal_to_table_idx[g], ix, iy, it]

    Multiple unique goals processed in parallel via multiprocessing.Pool.

Query convention (RL reward shaping)
-------------------------------------
Given continuous end-effector pose (x, y, theta) and goal index g:

    ix, iy  = world_to_cell((x, y))
    tidx    = goal_to_table_idx[g]
    phi     = np.min(H_all[tidx, ix, iy, :])   # min over θ-bins (paper smoothing)
    reward  = r_task + phi_next - phi_curr

    H value is inf if state is unreachable from goal — treat as large penalty or clip.

Requirements
------------
    pip install numpy cupy-cuda12x tqdm
    (use cupy-cuda11x for CUDA 11.x — check: nvcc --version)

Usage
-----
    python generate_lookup_tables_gpu.py                  # full run
    python generate_lookup_tables_gpu.py --skip-arc-precompute
    python generate_lookup_tables_gpu.py --merge-only
"""

import argparse
import heapq
import os
import time
from math import hypot

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GOALS_FILE     = os.path.join(os.path.dirname(__file__),
                 "tdcr_sim_mujoco/exploration_data/explored_configs_5cyl_labelled.npz")
OUT_DIR        = "lookup_tables_output"
FINAL_OUT      = "lookup_tables_5cyl_all_goals.npz"
ARC_CACHE_FILE = "arc_cache.npz"

CONTACT_MODE = "all"    # "all" | "contact" | "free"
MAX_GOALS    = None     # None = all raw goals; set int to cap for testing
N_WORKERS    = 14       # parallel Dijkstra workers

# ─────────────────────────────────────────────────────────────────────────────
# WORLD  (must match your simulation exactly)
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(11)

xlim = (-0.272, 0.268)
ylim = (-0.094, 0.376)

cell_size = 0.0075
Ntheta    = 24
dtheta    = 2 * np.pi / Ntheta

robot_r      = 0.001
pen_reject   = 0.002
contact_band = 0.01
ARC_SAMPLES_PER_M = 120

length_weight   = 1.0
lambda_progress = 6.0

start_in    = np.array([0.0, 0.1], dtype=float)
start_theta = np.pi / 2

Nx = int(np.ceil((xlim[1] - xlim[0]) / cell_size))
Ny = int(np.ceil((ylim[1] - ylim[0]) / cell_size))

circles = np.array([
    (-0.10,  0.15, 0.015),
    (-0.045, 0.26, 0.015),
    ( 0.02,  0.15, 0.015),
    ( 0.14,  0.15, 0.015),
    ( 0.08,  0.26, 0.015),
], dtype=np.float64)

# ─────────────────────────────────────────────────────────────────────────────
# MOTION PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────
KAPPA_FREE = np.array([-15., -10., -5., 0., 5., 10., 15.], dtype=np.float64)
S_FREE     = np.array([-0.02, -0.01, 0.01, 0.02],          dtype=np.float64)

KAPPA_CONTACT = np.array([-20.,-15.,-10.,-7.5,-5.,-2.5, 0.,
                            2.5,  5.,  7.5, 10., 15., 20.], dtype=np.float64)
S_CONTACT     = np.array([-0.02,-0.015,-0.01,-0.005,
                            0.005, 0.01, 0.015, 0.02],       dtype=np.float64)

CONTACT_SLIDE_COST_SCALE  = 1.08
CONTACT_PIVOT_COST_SCALE  = 1.15
CONTACT_PIVOT_ORBIT_GAIN  = 0.6
CONTACT_ALLOW_INSIDE_BAND = 0.0015

# ─────────────────────────────────────────────────────────────────────────────
# CPU HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def theta_from_bin(it):
    return wrap_angle(-np.pi + (it + 0.5) * dtheta)

def bin_from_theta(theta):
    theta = wrap_angle(theta)
    return int(np.floor((theta + np.pi) / dtheta)) % Ntheta

def cell_center(ix, iy):
    return xlim[0] + (ix + 0.5) * cell_size, ylim[0] + (iy + 0.5) * cell_size

def circular_bin_dist(a, b, N):
    d = abs(a - b)
    return min(d, N - d)

def world_to_cell(p):
    ix = int((float(p[0]) - xlim[0]) / cell_size)
    iy = int((float(p[1]) - ylim[0]) / cell_size)
    return max(0, min(Nx-1, ix)), max(0, min(Ny-1, iy))

def _signed_dist_cpu(x, y):
    dmin = np.inf
    for cx, cy, r in circles:
        d = hypot(x - cx, y - cy) - (r + robot_r)
        if d < dmin:
            dmin = d
    return dmin

def _is_contact_like_cpu(x, y):
    d = _signed_dist_cpu(x, y)
    return -CONTACT_ALLOW_INSIDE_BAND <= d <= contact_band

def _integrate_cc_cpu(x, y, theta, kappa, s):
    if abs(kappa) < 1e-12:
        c, ss = np.cos(theta), np.sin(theta)
        return x + c*s, y + ss*s, theta
    dth = kappa * s
    dx  = np.sin(dth) / kappa
    dy  = (1.0 - np.cos(dth)) / kappa
    c, ss = np.cos(theta), np.sin(theta)
    return x + c*dx - ss*dy, y + ss*dx + c*dy, wrap_angle(theta + dth)

def _nearest_obstacle_cpu(x, y):
    best, best_d = None, np.inf
    for cx, cy, r in circles:
        dx, dy = x - cx, y - cy
        dist = hypot(dx, dy)
        if dist < 1e-9:
            continue
        if dist < best_d:
            best_d = dist
            nx_, ny_ = dx / dist, dy / dist
            best = {"cx": cx, "cy": cy, "r": r,
                    "normal":  np.array([nx_, ny_]),
                    "tangent": np.array([-ny_, nx_])}
    return best

def _rotate_cpu(px, py, cx, cy, ang):
    dx, dy = px - cx, py - cy
    ca, sa = np.cos(ang), np.sin(ang)
    return cx + ca*dx - sa*dy, cy + sa*dx + ca*dy

def _validate_segment_cpu(x1, y1, x2, y2, samples=12):
    for i in range(samples):
        t = i / max(1, samples - 1)
        d = _signed_dist_cpu((1-t)*x1 + t*x2, (1-t)*y1 + t*y2)
        if max(0., -d) > pen_reject:
            return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# CONTACT ACTIONS
# ─────────────────────────────────────────────────────────────────────────────
def _make_contact_actions():
    a = []
    for ds in [0.003, 0.006, 0.01]:
        a += [("slide_t+", ds, 0), ("slide_t-", ds, 0)]
    for ds in [0.003, 0.006]:
        for dth in [-1, 1]:
            a += [("slide_turn_t+", ds, dth), ("slide_turn_t-", ds, dth)]
    for dth in [-2, -1, 1, 2]:
        a.append(("pivot_contact", 0., dth))
    return a

CONTACT_ACTIONS = _make_contact_actions()

def _apply_contact_action_cpu(x, y, theta, action, obs):
    kind, ds, dth_bins = action
    t = obs["tangent"]
    if kind == "slide_t+":
        return x + ds*t[0], y + ds*t[1], theta
    if kind == "slide_t-":
        return x - ds*t[0], y - ds*t[1], theta
    if kind == "slide_turn_t+":
        return x + ds*t[0], y + ds*t[1], wrap_angle(theta + dth_bins*dtheta)
    if kind == "slide_turn_t-":
        return x - ds*t[0], y - ds*t[1], wrap_angle(theta + dth_bins*dtheta)
    if kind == "pivot_contact":
        ang = float(dth_bins) * dtheta * CONTACT_PIVOT_ORBIT_GAIN
        x2, y2 = _rotate_cpu(x, y, obs["cx"], obs["cy"], ang)
        return x2, y2, wrap_angle(theta + dth_bins*dtheta)
    raise ValueError(kind)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: GPU ARC PRECOMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def precompute_arcs_gpu(out_path=ARC_CACHE_FILE):
    """
    Compute arc validity for every (ix, iy, it, kappa, s) on GPU.
    Saves:
      arc_valid_free    shape (Nx, Ny, Ntheta, n_kf, n_sf)
      arc_valid_contact shape (Nx, Ny, Ntheta, n_kc, n_sc)
    """
    try:
        import cupy as cp
    except ImportError:
        print("CuPy not found — falling back to CPU arc precomputation (slow).")
        return precompute_arcs_cpu(out_path)

    print(f"\n{'='*60}")
    print(f"Phase 1: GPU arc precomputation  [{Nx}×{Ny}×{Ntheta}, {len(circles)} obstacles]")
    t0 = time.time()

    xs     = xlim[0] + (np.arange(Nx) + 0.5) * cell_size
    ys     = ylim[0] + (np.arange(Ny) + 0.5) * cell_size
    thetas = np.array([wrap_angle(-np.pi + (it+0.5)*dtheta) for it in range(Ntheta)])

    cxs_g = cp.array(circles[:, 0], dtype=cp.float32)
    cys_g = cp.array(circles[:, 1], dtype=cp.float32)
    crs_g = cp.array(circles[:, 2] + robot_r, dtype=cp.float32)

    def _gpu_arc_valid(kappas, ss_arr):
        nk, ns   = len(kappas), len(ss_arr)
        n_steps  = max(2, int(max(abs(ss_arr)) * ARC_SAMPLES_PER_M) + 1)
        shape    = (Nx, Ny, Ntheta, nk, ns)

        X = cp.array(xs,     dtype=cp.float32)[:, None, None, None, None]
        Y = cp.array(ys,     dtype=cp.float32)[None, :, None, None, None]
        T = cp.array(thetas, dtype=cp.float32)[None, None, :, None, None]
        K = cp.array(kappas, dtype=cp.float32)[None, None, None, :, None]
        S = cp.array(ss_arr, dtype=cp.float32)[None, None, None, None, :]

        valid = cp.ones(shape, dtype=cp.bool_)

        for step_i in range(n_steps):
            st      = (step_i / (n_steps - 1)) * S
            abs_k   = cp.abs(K)
            dth     = K * st
            sin_dth = cp.sin(dth)
            cos_dth = cp.cos(dth)

            dx_loc = cp.where(abs_k > 1e-6, sin_dth / (K + 1e-30), st)
            dy_loc = cp.where(abs_k > 1e-6, (1.0 - cos_dth) / (K + 1e-30),
                              cp.zeros_like(st))

            xt = X + cp.cos(T)*dx_loc - cp.sin(T)*dy_loc
            yt = Y + cp.sin(T)*dx_loc + cp.cos(T)*dy_loc

            d_min = cp.full(shape, cp.inf, dtype=cp.float32)
            for j in range(len(circles)):
                d_j   = cp.sqrt((xt - cxs_g[j])**2 + (yt - cys_g[j])**2) - crs_g[j]
                d_min = cp.minimum(d_min, d_j)

            valid &= (cp.maximum(0.0, -d_min) <= pen_reject)

        return cp.asnumpy(valid)

    print("  Free-space arcs...")
    t1 = time.time()
    arc_free = _gpu_arc_valid(KAPPA_FREE, S_FREE)
    print(f"  Done {time.time()-t1:.1f}s  valid: {arc_free.sum():,}/{arc_free.size:,}")

    print("  Contact arcs...")
    t1 = time.time()
    arc_contact = _gpu_arc_valid(KAPPA_CONTACT, S_CONTACT)
    print(f"  Done {time.time()-t1:.1f}s  valid: {arc_contact.sum():,}/{arc_contact.size:,}")

    np.savez_compressed(out_path,
        arc_valid_free    = arc_free,
        arc_valid_contact = arc_contact,
        kappa_free        = KAPPA_FREE,
        s_free            = S_FREE,
        kappa_contact     = KAPPA_CONTACT,
        s_contact         = S_CONTACT,
    )
    print(f"  Saved → {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"  GPU phase total: {time.time()-t0:.1f}s")


def precompute_arcs_cpu(out_path=ARC_CACHE_FILE):
    """CPU fallback — correct but slow (~hours for full grid)."""
    print("CPU arc precomputation (slow — consider installing CuPy)...")
    t0 = time.time()

    def _arc_valid_np(kappas, ss_arr):
        nk, ns = len(kappas), len(ss_arr)
        result = np.ones((Nx, Ny, Ntheta, nk, ns), dtype=bool)
        for ix in range(Nx):
            x = xlim[0] + (ix + 0.5) * cell_size
            for iy in range(Ny):
                y = ylim[0] + (iy + 0.5) * cell_size
                for it in range(Ntheta):
                    theta = wrap_angle(-np.pi + (it+0.5)*dtheta)
                    for ik, kappa in enumerate(kappas):
                        for is_, s in enumerate(ss_arr):
                            steps = max(2, int(abs(s)*ARC_SAMPLES_PER_M)+1)
                            for step_i in range(steps):
                                st = step_i/(steps-1)*s
                                xt, yt, _ = _integrate_cc_cpu(x, y, theta, kappa, st)
                                if max(0., -_signed_dist_cpu(xt, yt)) > pen_reject:
                                    result[ix, iy, it, ik, is_] = False
                                    break
        return result

    arc_free    = _arc_valid_np(KAPPA_FREE,    S_FREE)
    arc_contact = _arc_valid_np(KAPPA_CONTACT, S_CONTACT)
    np.savez_compressed(out_path,
        arc_valid_free=arc_free, arc_valid_contact=arc_contact,
        kappa_free=KAPPA_FREE, s_free=S_FREE,
        kappa_contact=KAPPA_CONTACT, s_contact=S_CONTACT)
    print(f"Arc cache saved in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# SUCCESSOR / REVERSE GRAPH  (reads arc_cache — no GPU dependency)
# ─────────────────────────────────────────────────────────────────────────────
_ARC_CACHE = None

def _load_arc_cache():
    global _ARC_CACHE
    if _ARC_CACHE is None:
        data = np.load(ARC_CACHE_FILE)
        _ARC_CACHE = {
            "free":    data["arc_valid_free"],
            "contact": data["arc_valid_contact"],
        }
    return _ARC_CACHE

def successors_from_cache(u, goal_xy):
    ix, iy, it = u
    x, y = cell_center(ix, iy)

    if max(0., -_signed_dist_cpu(x, y)) > pen_reject:
        return []

    near_contact = _is_contact_like_cpu(x, y)
    cache        = _load_arc_cache()
    arc_valid    = cache["contact"] if near_contact else cache["free"]
    KAPPA_SET    = KAPPA_CONTACT    if near_contact else KAPPA_FREE
    S_SET        = S_CONTACT        if near_contact else S_FREE
    theta        = theta_from_bin(it)
    xg, yg       = goal_xy
    out          = []

    # Constant-curvature primitives (validated by arc cache)
    for ik, kappa in enumerate(KAPPA_SET):
        for is_, s in enumerate(S_SET):
            if not arc_valid[ix, iy, it, ik, is_]:
                continue
            x2, y2, th2 = _integrate_cc_cpu(x, y, theta, float(kappa), float(s))
            if not (xlim[0] <= x2 <= xlim[1] and ylim[0] <= y2 <= ylim[1]):
                continue
            ix2, iy2 = world_to_cell((x2, y2))
            it2      = bin_from_theta(th2)
            if (ix2, iy2, it2) == (ix, iy, it):
                continue
            if max(0., -_signed_dist_cpu(*cell_center(ix2, iy2))) > pen_reject:
                continue
            progress = lambda_progress * max(0.,
                hypot(x2 - xg, y2 - yg) - hypot(x - xg, y - yg))
            out.append(((ix2, iy2, it2), max(1e-6, length_weight*abs(s) + progress)))

    # Contact slide/pivot primitives
    if near_contact:
        obs = _nearest_obstacle_cpu(x, y)
        if obs is not None:
            for action in CONTACT_ACTIONS:
                x2, y2, th2 = _apply_contact_action_cpu(x, y, theta, action, obs)
                if not (xlim[0] <= x2 <= xlim[1] and ylim[0] <= y2 <= ylim[1]):
                    continue
                if not _validate_segment_cpu(x, y, x2, y2):
                    continue
                ix2, iy2 = world_to_cell((x2, y2))
                it2      = bin_from_theta(th2)
                if (ix2, iy2, it2) == (ix, iy, it):
                    continue
                if max(0., -_signed_dist_cpu(*cell_center(ix2, iy2))) > pen_reject:
                    continue
                kind, ds, dth_bins = action
                spatial  = hypot(x2 - x, y2 - y)
                ang_cost = 0.003 * abs(dth_bins)
                scale    = (CONTACT_SLIDE_COST_SCALE if kind.startswith("slide")
                            else CONTACT_PIVOT_COST_SCALE)
                progress = lambda_progress * max(0.,
                    hypot(x2 - xg, y2 - yg) - hypot(x - xg, y - yg))
                out.append(((ix2, iy2, it2),
                             max(1e-6, scale*(spatial + ang_cost) + progress)))
    return out

def build_reverse_graph(goal_xy):
    rev = {}
    for ix in range(Nx):
        for iy in range(Ny):
            for it in range(Ntheta):
                u = (ix, iy, it)
                for v, w in successors_from_cache(u, goal_xy):
                    rev.setdefault(v, []).append((u, w))
    return rev

# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD DIJKSTRA
# ─────────────────────────────────────────────────────────────────────────────

def backward_dijkstra(goal_xy_cell, goal_theta_bin, rev, goal_theta_tol_bins=0):
    H  = np.full((Nx, Ny, Ntheta), np.inf, dtype=np.float32)
    gx, gy = goal_xy_cell
    pq = []
    for gt in range(Ntheta):
        if circular_bin_dist(gt, goal_theta_bin, Ntheta) <= goal_theta_tol_bins:
            H[gx, gy, gt] = 0.0
            pq.append((0.0, (gx, gy, gt)))
    heapq.heapify(pq)
    closed = set()
    while pq:
        hcur, c = heapq.heappop(pq)
        if c in closed:
            continue
        closed.add(c)
        for u, w in rev.get(c, []):
            ux, uy, ut = u
            nh = hcur + w
            if nh < H[ux, uy, ut]:
                H[ux, uy, ut] = nh
                heapq.heappush(pq, (nh, u))
    return H

# ─────────────────────────────────────────────────────────────────────────────
# GOAL LOADING + DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_deduplicate_goals(npz_path, contact_mode="all", max_goals=None):
    """
    Load all raw goals and deduplicate to unique (ix, iy, theta_bin) keys.

    Returns
    -------
    raw_goals_xy     : (N_raw, 2)  — all original goal positions
    raw_goal_thetas  : (N_raw,)    — all original goal headings
    unique_goals     : list of dict, one per unique (ix, iy, theta_bin)
        Each dict: {key, ix, iy, gt_bin, goal_xy, goal_theta,
                    raw_indices}  ← list of raw indices that map here
    goal_to_table_idx: (N_raw,) int array — index into unique_goals for each raw goal
    """
    data = np.load(npz_path, allow_pickle=True)
    tip_positions = np.asarray(data["tip_positions"], dtype=float)
    tip_poses     = np.asarray(data["tip_poses"],     dtype=float)
    goals_xy      = tip_positions[:, :2].copy()
    R_all         = tip_poses.reshape(-1, 3, 3)
    goal_thetas   = np.array(
        [wrap_angle(np.arctan2(R[1, 0], R[0, 0])) for R in R_all], dtype=float)

    mask = np.ones(len(goals_xy), dtype=bool)
    if contact_mode != "all":
        cr   = np.asarray(data["contact_required"]).astype(bool)
        mask = cr if contact_mode == "contact" else ~cr

    goals_xy    = goals_xy[mask]
    goal_thetas = goal_thetas[mask]
    raw_indices_all = np.where(mask)[0]  # original indices before masking

    if max_goals is not None:
        goals_xy    = goals_xy[:max_goals]
        goal_thetas = goal_thetas[:max_goals]
        raw_indices_all = raw_indices_all[:max_goals]

    N_raw = len(goals_xy)
    print(f"Loaded {N_raw} raw goals (contact_mode='{contact_mode}')")

    # Snap every raw goal to grid
    cells = np.array([world_to_cell(goals_xy[i]) for i in range(N_raw)])   # (N,2)
    tbins = np.array([bin_from_theta(goal_thetas[i]) for i in range(N_raw)], dtype=int)

    # Build dedup map: (ix, iy, tbin) → unique table index
    key_to_uidx = {}
    unique_goals = []
    goal_to_table_idx = np.zeros(N_raw, dtype=np.int32)

    for i in range(N_raw):
        ix, iy = int(cells[i, 0]), int(cells[i, 1])
        gt     = int(tbins[i])
        key    = (ix, iy, gt)
        if key not in key_to_uidx:
            uidx = len(unique_goals)
            key_to_uidx[key] = uidx
            gx_c, gy_c = cell_center(ix, iy)
            unique_goals.append({
                "key":         key,
                "ix":          ix,
                "iy":          iy,
                "gt_bin":      gt,
                "goal_xy":     np.array([gx_c, gy_c]),
                "goal_theta":  wrap_angle(float(goal_thetas[i])),
                "raw_indices": [],
            })
        uidx = key_to_uidx[key]
        unique_goals[uidx]["raw_indices"].append(i)
        goal_to_table_idx[i] = uidx

    n_unique = len(unique_goals)
    n_dup    = N_raw - n_unique
    print(f"Unique (cell, θ-bin) goal keys: {n_unique}  "
          f"({n_dup} raw goals are duplicates → mapped, not recomputed)")
    print(f"Deduplication ratio: {N_raw/n_unique:.1f}x")

    return goals_xy, goal_thetas, unique_goals, goal_to_table_idx

# ─────────────────────────────────────────────────────────────────────────────
# PER-UNIQUE-GOAL WORKER
# ─────────────────────────────────────────────────────────────────────────────

def process_unique_goal(args):
    """Run backward Dijkstra for one unique (ix, iy, θ-bin) goal."""
    uidx, goal_info, out_dir = args
    ckpt = os.path.join(out_dir, f"unique_{uidx:05d}.npz")
    if os.path.exists(ckpt):
        return uidx, True   # already done

    _load_arc_cache()

    ix, iy   = goal_info["ix"], goal_info["iy"]
    gt_bin   = goal_info["gt_bin"]
    goal_xy  = goal_info["goal_xy"]
    goal_node = (ix, iy, gt_bin)

    sx, sy     = world_to_cell(start_in)
    start_node = (sx, sy, bin_from_theta(start_theta))

    try:
        rev = build_reverse_graph(goal_xy)
        H   = backward_dijkstra((ix, iy), gt_bin, rev, goal_theta_tol_bins=0)

        # Reachability checks
        if not np.isfinite(H[start_node]):
            return uidx, False
        if not np.any(np.isfinite(H[ix, iy, :])):
            return uidx, False
        goal_slice = H[ix, iy, :]
        finite_mask = np.isfinite(goal_slice)
        if not finite_mask[gt_bin]:
            return uidx, False

        np.savez_compressed(ckpt,
            H          = H.astype(np.float32),
            ix         = np.int32(ix),
            iy         = np.int32(iy),
            gt_bin     = np.int32(gt_bin),
            goal_xy    = goal_xy.astype(np.float32),
            goal_theta = np.float32(goal_info["goal_theta"]),
            goal_node  = np.array(goal_node, dtype=np.int32),
            finite_count = np.int32(int(np.isfinite(H).sum())),
        )
        return uidx, True

    except Exception as e:
        print(f"  [unique goal {uidx}] ERROR: {e}", flush=True)
        return uidx, False

# ─────────────────────────────────────────────────────────────────────────────
# MERGE  →  final .npz
# ─────────────────────────────────────────────────────────────────────────────

def merge_checkpoints(out_dir, final_out, unique_goals, goal_to_table_idx,
                      raw_goals_xy, raw_goal_thetas):
    """
    Merge per-unique-goal checkpoints into one .npz.

    The output arrays are indexed by UNIQUE goal index (0..N_unique-1).
    Use goal_to_table_idx[g] to look up H for raw goal g:

        phi = np.min(H_all[goal_to_table_idx[g], ix, iy, :])
    """
    n_unique = len(unique_goals)
    H_list, xy_list, theta_list, tbin_list, node_list, fc_list = \
        [], [], [], [], [], []

    missing = []
    for uidx in range(n_unique):
        ckpt = os.path.join(out_dir, f"unique_{uidx:05d}.npz")
        if not os.path.exists(ckpt):
            missing.append(uidx)
            continue
        d = np.load(ckpt)
        H_list.append(d["H"])
        xy_list.append(d["goal_xy"])
        theta_list.append(float(d["goal_theta"]))
        tbin_list.append(int(d["gt_bin"]))
        node_list.append(d["goal_node"])
        fc_list.append(int(d["finite_count"]))

    if missing:
        print(f"  WARNING: {len(missing)} unique goals missing checkpoints "
              f"(failed reachability or errored). Their table entries will be absent.")

    if not H_list:
        raise RuntimeError(f"No valid checkpoints found in {out_dir}")

    # Build a compact table index — only successfully computed unique goals
    # Re-map goal_to_table_idx to point into the compact stack
    computed_uidxs = [uidx for uidx in range(n_unique)
                      if os.path.exists(os.path.join(out_dir, f"unique_{uidx:05d}.npz"))]
    compact_map = {uidx: ci for ci, uidx in enumerate(computed_uidxs)}

    # goal_to_table_idx for raw goals: -1 if their unique goal failed
    N_raw = len(goal_to_table_idx)
    final_goal_to_table_idx = np.full(N_raw, -1, dtype=np.int32)
    for raw_i, uidx in enumerate(goal_to_table_idx):
        if uidx in compact_map:
            final_goal_to_table_idx[raw_i] = compact_map[uidx]

    n_valid_raw = int((final_goal_to_table_idx >= 0).sum())
    sx, sy     = world_to_cell(start_in)
    start_node = (sx, sy, bin_from_theta(start_theta))

    np.savez_compressed(final_out,
        # ── Core lookup data ──────────────────────────────────────────────
        H_all                  = np.stack(H_list),          # (N_unique, Nx, Ny, Ntheta)
        goal_to_table_idx      = final_goal_to_table_idx,   # (N_raw,) → index into H_all

        # ── Goal metadata (indexed by unique goal) ────────────────────────
        unique_goal_positions  = np.stack(xy_list),         # (N_unique, 2)
        unique_goal_thetas     = np.array(theta_list, dtype=np.float32),
        unique_goal_theta_bins = np.array(tbin_list,  dtype=np.int32),
        unique_goal_nodes      = np.stack(node_list),       # (N_unique, 3)
        finite_counts          = np.array(fc_list,    dtype=np.int32),

        # ── Raw goal data (original order, all ~8000) ─────────────────────
        raw_goal_positions     = raw_goals_xy.astype(np.float32),
        raw_goal_thetas        = raw_goal_thetas.astype(np.float32),

        # ── Grid metadata ─────────────────────────────────────────────────
        start_position         = start_in.astype(np.float32),
        start_node             = np.array(start_node, dtype=np.int32),
        xlim                   = np.array(xlim,    dtype=np.float32),
        ylim                   = np.array(ylim,    dtype=np.float32),
        cell_size              = np.float32(cell_size),
        Ntheta                 = np.int32(Ntheta),
        dtheta                 = np.float32(dtheta),
        heuristic_type         = np.array(
            "RAO-LIKE contact-aware: CC arcs + contact slide/pivot, all goals",
            dtype=object),
    )

    H_shape = np.load(final_out)["H_all"].shape
    print(f"\nMerge complete:")
    print(f"  Unique goal tables:  {len(H_list)}/{n_unique}")
    print(f"  Raw goals covered:   {n_valid_raw}/{N_raw}")
    print(f"  H_all shape:         {H_shape}   "
          f"({np.prod(H_shape)*4/1e6:.0f} MB uncompressed)")
    print(f"  Output:              {final_out}")
    print(f"\nQuery usage in RL:")
    print(f"  data = np.load('{final_out}')")
    print(f"  H_all             = data['H_all']            # shape {H_shape}")
    print(f"  goal_to_table_idx = data['goal_to_table_idx']")
    print(f"  # For goal g, state (x, y):")
    print(f"  tidx = goal_to_table_idx[g]   # -1 → goal had no valid table")
    print(f"  phi  = np.min(H_all[tidx, ix, iy, :])  # min over θ-bins (paper smoothing)")

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: RL reward shaping helper (importable separately)
# ─────────────────────────────────────────────────────────────────────────────

class HeuristicTable:
    """
    Thin wrapper around the saved lookup table for use in RL training.

    Usage
    -----
        ht = HeuristicTable("lookup_tables_5cyl_all_goals.npz")
        phi = ht.phi(x, y, goal_idx)       # scalar potential Φ(s)
        r   = r_task + ht.phi(x2,y2,g) - ht.phi(x1,y1,g)   # shaped reward
    """
    INF_SUBSTITUTE = 1e6   # used when state is unreachable — tune to your reward scale

    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.H_all             = d["H_all"]               # (N_unique, Nx, Ny, Ntheta)
        self.goal_to_table_idx = d["goal_to_table_idx"]   # (N_raw,)
        self.xlim      = tuple(d["xlim"])
        self.ylim      = tuple(d["ylim"])
        self.cell_size = float(d["cell_size"])
        self.Nx        = self.H_all.shape[1]
        self.Ny        = self.H_all.shape[2]

    def _world_to_cell(self, x, y):
        ix = int((x - self.xlim[0]) / self.cell_size)
        iy = int((y - self.ylim[0]) / self.cell_size)
        return max(0, min(self.Nx-1, ix)), max(0, min(self.Ny-1, iy))

    def phi(self, x: float, y: float, goal_idx: int) -> float:
        """
        Potential Φ(s) for end-effector at (x, y) with goal goal_idx.
        Uses min over θ-bins as in Rao et al. Section III-D.
        Returns INF_SUBSTITUTE if goal has no table or state is unreachable.
        """
        tidx = int(self.goal_to_table_idx[goal_idx])
        if tidx < 0:
            return self.INF_SUBSTITUTE
        ix, iy = self._world_to_cell(x, y)
        h_slice = self.H_all[tidx, ix, iy, :]   # shape (Ntheta,)
        val = float(np.min(h_slice))
        return val if np.isfinite(val) else self.INF_SUBSTITUTE

    def shaped_reward(self, x1, y1, x2, y2, goal_idx, r_task=0.0):
        """r_task + Φ(s_{t+1}) - Φ(s_t)"""
        return r_task + self.phi(x2, y2, goal_idx) - self.phi(x1, y1, goal_idx)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(skip_arc_precompute=False, merge_only=False):
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Phase 1: GPU arc precomputation ──────────────────────────────────────
    if not merge_only:
        if not skip_arc_precompute and not os.path.exists(ARC_CACHE_FILE):
            precompute_arcs_gpu(ARC_CACHE_FILE)
        elif os.path.exists(ARC_CACHE_FILE):
            print(f"Arc cache found at {ARC_CACHE_FILE} — skipping GPU phase.")
        else:
            print("No cache found — running GPU phase.")
            precompute_arcs_gpu(ARC_CACHE_FILE)

        print("Loading arc cache into main process...")
        _load_arc_cache()
        print(f"  free    shape: {_ARC_CACHE['free'].shape}")
        print(f"  contact shape: {_ARC_CACHE['contact'].shape}")

    # ── Load + deduplicate goals ──────────────────────────────────────────────
    raw_goals_xy, raw_goal_thetas, unique_goals, goal_to_table_idx = \
        load_and_deduplicate_goals(GOALS_FILE, CONTACT_MODE, MAX_GOALS)

    n_unique = len(unique_goals)
    n_raw    = len(raw_goals_xy)

    if merge_only:
        merge_checkpoints(OUT_DIR, FINAL_OUT, unique_goals,
                          goal_to_table_idx, raw_goals_xy, raw_goal_thetas)
        return

    # ── Phase 2: per-unique-goal Dijkstra ────────────────────────────────────
    done = sum(1 for i in range(n_unique)
               if os.path.exists(os.path.join(OUT_DIR, f"unique_{i:05d}.npz")))
    print(f"\nGrid: {Nx} × {Ny} × {Ntheta} = {Nx*Ny*Ntheta:,} states")
    print(f"Unique goals: {n_unique}  |  Already done: {done}  |  Workers: {N_WORKERS}")
    print(f"(Raw goals: {n_raw}, saved via goal_to_table_idx mapping)\n")

    job_args = [(i, unique_goals[i], OUT_DIR) for i in range(n_unique)]

    try:
        from tqdm import tqdm
        bar = tqdm(total=n_unique, initial=done, unit="goal", desc="Dijkstra")
        def tick(_): bar.update(1)
    except ImportError:
        bar = None
        def tick(i): print(f"  unique goal {i}/{n_unique}", flush=True)

    stored = done
    t_start = time.time()

    if N_WORKERS > 1:
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=N_WORKERS) as pool:
            for uidx, success in pool.imap_unordered(process_unique_goal, job_args):
                tick(uidx)
                if success:
                    stored += 1
    else:
        for args in job_args:
            uidx, success = process_unique_goal(args)
            tick(uidx)
            if success:
                stored += 1

    if bar:
        bar.close()

    elapsed  = time.time() - t_start
    per_goal = elapsed / max(1, n_unique - done)
    print(f"\nDone. {stored}/{n_unique} unique goals computed in {elapsed:.1f}s "
          f"({per_goal:.1f}s/goal avg)")

    print("\nMerging checkpoints...")
    merge_checkpoints(OUT_DIR, FINAL_OUT, unique_goals,
                      goal_to_table_idx, raw_goals_xy, raw_goal_thetas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-arc-precompute", action="store_true")
    parser.add_argument("--merge-only",          action="store_true")
    args = parser.parse_args()
    main(skip_arc_precompute=args.skip_arc_precompute,
         merge_only=args.merge_only)