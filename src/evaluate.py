"""
Fair evaluation of FROZEN agents on identical games.

    rows = evaluate(agent, seeds("test", 100))      # one dict per game
    print(summarize(rows))                          # win rate + 95% CI, score difference, ...
    print(paired_compare(rows_a, rows_b))           # A vs B on the SAME seeds

Scoring of one game (what we record)
    agent_pts, ai_pts   points of the agent / of the built-in AI (a game ends at 10)
    won / lost          1 if the agent / the AI reached 10 first
    truncated           1 if neither reached 10 within max_steps (counts as not won)
    diff                agent_pts - ai_pts
    paper_reward        +1 per own point, -2 per point conceded, +20 for winning, -20 for losing
                        (the reward scheme of the Stanford paper, so curves are comparable)
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pong_env import PongEnv  # noqa: E402

# Three disjoint sets of game seeds. Train on one, choose checkpoints on "val",
# and look at "test" only for the final numbers.
SEED_SETS = {
    "train": range(0, 9_999),
    "val": range(10_000, 10_100),
    "test": range(20_000, 20_100),
}


def seeds(split="test", n=None):
    s = list(SEED_SETS[split])
    return s[:n] if n else s


def paper_reward(agent_pts, ai_pts, won, lost):
    return agent_pts - 2 * ai_pts + (20 if won else -20 if lost else 0)


# ----------------------------------------------------------------------------- playing
def play_game(agent, seed, rom=None, max_steps=30_000):
    env = PongEnv(rom, seed=seed, hit_reward=0.0, max_steps=max_steps)
    obs = env.reset()
    agent.reset(seed)
    done, info, steps = False, {}, 0
    while not done:
        obs, _, done, info = env.step(agent.act(obs))
        steps += 1
    a, b = info["agent_score"], info["ai_score"]
    won = int(info["won"])
    lost = int(b >= 10 and not won)
    return {"seed": seed, "agent_pts": a, "ai_pts": b, "steps": steps, "won": won, "lost": lost,
            "truncated": int(info["truncated"]), "diff": a - b,
            "paper_reward": paper_reward(a, b, won, lost)}


def evaluate(agent, seed_list, rom=None, max_steps=30_000, progress=None):
    rows = []
    for i, s in enumerate(seed_list, 1):
        rows.append(play_game(agent, s, rom, max_steps))
        if progress:
            progress(i, len(seed_list))
    return rows


# ----------------------------------------------------------------------------- statistics
def wilson(wins, n, z=1.96):
    """95% confidence interval for a win rate (works for small n and for 0% / 100%)."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def summarize(rows):
    n = len(rows)
    wins = sum(r["won"] for r in rows)
    lo, hi = wilson(wins, n)
    m = lambda k: float(np.mean([r[k] for r in rows])) if rows else 0.0
    return {"games": n, "wins": wins, "win_rate": wins / n if n else 0.0, "ci_low": lo, "ci_high": hi,
            "agent_pts": m("agent_pts"), "ai_pts": m("ai_pts"), "diff": m("diff"),
            "steps": m("steps"), "paper_reward": m("paper_reward"),
            "truncated": sum(r["truncated"] for r in rows)}


def _binom_two_sided(k, n):
    """Exact two-sided p-value of observing k (or more extreme) out of n fair coin flips."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_compare(rows_a, rows_b, n_boot=4000, seed=0):
    """Compare two agents that played the SAME seeds.

    Outcome table per seed:  both win / only A wins / only B wins / both lose.
    Only the 'discordant' seeds (one wins, the other not) carry information about which agent is
    better; an exact sign test (McNemar) says whether the imbalance could be luck.
    Score difference: mean over seeds of (diff_A - diff_B) with a paired bootstrap 95% interval.
    """
    a = {r["seed"]: r for r in rows_a}
    b = {r["seed"]: r for r in rows_b}
    common = sorted(set(a) & set(b))
    both = sum(a[s]["won"] and b[s]["won"] for s in common)
    only_a = sum(a[s]["won"] and not b[s]["won"] for s in common)
    only_b = sum(b[s]["won"] and not a[s]["won"] for s in common)
    neither = len(common) - both - only_a - only_b
    p = _binom_two_sided(min(only_a, only_b), only_a + only_b)
    d = np.array([a[s]["diff"] - b[s]["diff"] for s in common], dtype=float)
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)] if len(d) else [0.0]
    return {"games": len(common), "both_win": both, "only_a": only_a, "only_b": only_b, "neither": neither,
            "p_value": p, "mean_diff_gap": float(d.mean()) if len(d) else 0.0,
            "gap_ci_low": float(np.percentile(boots, 2.5)), "gap_ci_high": float(np.percentile(boots, 97.5))}


# ----------------------------------------------------------------------------- output
def write_csv(rows, path, extra=None):
    """rows: list of dicts. extra: constant columns added to every row, e.g. {'agent': 'track'}."""
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = [({**(extra or {}), **r}) for r in rows]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def format_table(named_summaries):
    head = f"{'agent':28s} {'games':>5s} {'wins':>5s} {'win%':>6s} {'95% CI':>13s} {'pts me-AI':>11s} {'diff':>6s} {'steps':>7s} {'paper R':>8s} {'trunc':>5s}"
    lines = [head, "-" * len(head)]
    for name, s in named_summaries:
        lines.append(f"{name[:28]:28s} {s['games']:5d} {s['wins']:5d} {100 * s['win_rate']:5.0f}% "
                     f"{100 * s['ci_low']:5.0f}-{100 * s['ci_high']:3.0f}% "
                     f"{s['agent_pts']:5.1f}-{s['ai_pts']:<5.1f} {s['diff']:+6.1f} {s['steps']:7.0f} "
                     f"{s['paper_reward']:8.1f} {s['truncated']:5d}")
    return "\n".join(lines)
