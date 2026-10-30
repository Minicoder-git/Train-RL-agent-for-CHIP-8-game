"""
Every player the benchmark can evaluate, behind ONE interface:

    agent = make_agent("track")        # or "random", "idle", "top", "bottom", "smart",
    agent.reset(seed)                  #    "vi:models/pong_vi.npz", "dqn:models/pong_dqn.npz"
    action = agent.act(obs)            # obs from PongEnv, action 0 = stay, 1 = up, 2 = down

Agents never learn inside the benchmark: they are FROZEN policies. Learning happens beforehand
(train_dqn.py / tools/make_vi_baseline.py) and produces a file; the benchmark only loads it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STAY, UP, DOWN = 0, 1, 2


class Agent:
    name = "agent"

    def reset(self, seed=0):
        """Called before every game. Agents with their own randomness re-seed here."""

    def act(self, obs):
        raise NotImplementedError


# ----------------------------------------------------------------------------- B0, B1
class RandomAgent(Agent):
    name = "random"

    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return int(self.rng.integers(3))


class IdleAgent(Agent):
    name = "idle"

    def act(self, obs):
        return STAY


# ----------------------------------------------------------------------------- B2 heuristics
class TrackAgent(Agent):
    """Keeps the ball level with a chosen spot on the paddle. No learning, just a rule.

    shift = where on the paddle we want to meet the ball, relative to the paddle centre (pixels):
        0 -> centre (ball goes straight), -2 -> upper edge (ball bends up), +2 -> lower edge (bends down)
    'smart' aims for the edge that sends the ball AWAY from the AI paddle.
    """

    def __init__(self, name="track", shift=0.0, smart=False):
        self.name, self.shift, self.smart = name, shift, smart

    def act(self, obs):
        dy = obs[5] * 31                                   # ball y - paddle centre, in pixels
        shift = self.shift
        if self.smart:
            ai_centre = obs[6] * 31
            shift = 2.0 if ai_centre < 16 else -2.0
        d = dy - shift
        if d < -1.0:
            return UP
        if d > 1.0:
            return DOWN
        return STAY


# ----------------------------------------------------------------------------- B3 value iteration
DY_BINS, X_BINS = 13, 8
N_STATES = DY_BINS * X_BINS * 2 * 3                         # + 1 absorbing "game over" state


def vi_state(obs):
    """Discretise an observation: (ball-to-paddle offset in 4 px bins, ball x in 8 bins, vx sign, vy)."""
    dy = int(np.clip(round(obs[5] * 31 / 4.0), -6, 6)) + 6
    xb = min(X_BINS - 1, int(obs[0] * 63) // 8)
    vx = 1 if obs[2] > 0 else 0
    vy = int(round(obs[3])) + 1
    return ((dy * X_BINS + xb) * 2 + vx) * 3 + vy


class ValueIterationAgent(Agent):
    name = "vi"

    def __init__(self, path):
        d = np.load(path)
        self.policy = d["policy"]

    def act(self, obs):
        return int(self.policy[vi_state(obs)])


# ----------------------------------------------------------------------------- learned DQN
class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, path):
        from train_dqn import QNet
        self.net = QNet.load(path)

    def act(self, obs):
        return self.net.act(obs)


# ----------------------------------------------------------------------------- factory
def make_agent(spec):
    """'random' | 'idle' | 'track' | 'top' | 'bottom' | 'smart' | 'vi:<file.npz>' | 'dqn:<file.npz>'"""
    kind, _, arg = spec.partition(":")
    kind = kind.lower()
    if kind == "random":
        return RandomAgent()
    if kind == "idle":
        return IdleAgent()
    if kind == "track":
        return TrackAgent("track", 0.0)
    if kind == "top":
        return TrackAgent("top", -2.0)
    if kind == "bottom":
        return TrackAgent("bottom", 2.0)
    if kind == "smart":
        return TrackAgent("smart", smart=True)
    if kind in ("vi", "dqn"):
        if not arg:
            raise ValueError(f"'{kind}' needs a file, e.g. {kind}:models/pong_{kind}.npz")
        a = ValueIterationAgent(arg) if kind == "vi" else DQNAgent(arg)
        a.name = f"{kind}:{os.path.basename(arg)}"
        return a
    raise ValueError(f"unknown agent '{spec}'")
