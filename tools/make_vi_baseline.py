"""
Build the value-iteration BASELINE (the paper's "discretised states + value iteration" idea).

    py tools/make_vi_baseline.py                  # -> models/pong_vi.npz
    py tools/make_vi_baseline.py --steps 400000

Steps
  1. Play uniformly random actions for --steps game steps and record every transition
     (state, action) -> next state, plus the reward.
  2. State = coarse bins of what the agent sees (ball height relative to the paddle in 4 px bins,
     ball x in 8 bins, ball direction, ball vertical speed): 624 states + one "game over" state.
  3. Estimate P(next state | state, action) and the mean reward from the counts.
  4. Run value iteration on that estimated model and keep the best action per state.

No neural network and no learning during play: the saved table IS the policy.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.join(HERE, "src"), os.path.join(HERE, "..", "src")):
    if os.path.isdir(cand):
        SRC = os.path.abspath(cand)
        sys.path.insert(0, SRC)
        break
else:
    sys.exit("cannot find the src/ folder next to this script")

from agents import N_STATES, vi_state  # noqa: E402  # type: ignore[reportMissingImports]
from pong_env import PongEnv  # noqa: E402 # type: ignore[reportMissingImports]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=200_000, help="random-play steps used to estimate the model")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--hit-reward", type=float, default=0.1, help="same shaping as the DQN default; 0 = points only")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rom", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(SRC), "models", "pong_vi.npz"))
    a = ap.parse_args()

    TERM = N_STATES
    counts = np.zeros((N_STATES, 3, N_STATES + 1))
    rsum = np.zeros((N_STATES, 3))
    rng = np.random.default_rng(a.seed)
    env = PongEnv(a.rom, seed=a.seed, hit_reward=a.hit_reward, max_steps=6000)
    s = vi_state(env.reset())
    for t in range(a.steps):
        act = int(rng.integers(3))
        obs, r, done, info = env.step(act)
        s2 = TERM if (done and not info["truncated"]) else vi_state(obs)
        counts[s, act, s2] += 1
        rsum[s, act] += r
        s = vi_state(env.reset()) if done else s2
    print(f"collected {a.steps:,} random transitions")

    n_sa = counts.sum(axis=2)                                   # visits of each (state, action)
    seen = n_sa > 0
    P = np.zeros_like(counts)
    P[seen] = counts[seen] / n_sa[seen][:, None]
    for st, ac in zip(*np.where(~seen)):                       # never tried: assume "nothing happens"
        P[st, ac, st] = 1.0
    R = np.where(seen, rsum / np.maximum(n_sa, 1), 0.0)

    V = np.zeros(N_STATES + 1)
    for it in range(1, 5001):
        Q = R + a.gamma * (P @ V)
        newV = np.append(Q.max(axis=1), 0.0)                   # game-over state keeps value 0
        delta = np.abs(newV - V).max()
        V = newV
        if delta < 1e-7:
            break
    policy = Q.argmax(axis=1).astype(np.int64)
    states_seen = int((n_sa.sum(axis=1) > 0).sum())
    print(f"value iteration converged after {it} sweeps (max change {delta:.1e})")
    print(f"states visited at least once: {states_seen}/{N_STATES}; (state,action) pairs seen: {int(seen.sum())}/{N_STATES * 3}")
    print("action choice counts  stay/up/down:", np.bincount(policy, minlength=3).tolist())

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, policy=policy, V=V, gamma=a.gamma)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
