"""
Double-DQN agent for the CHIP-8 Pong environment. Pure numpy: no PyTorch / TensorFlow needed.

    python src/train_dqn.py                          # train (Ctrl+C to stop, best model is kept)
    python src/train_dqn.py --episodes 300 --seed 1
    python src/train_dqn.py --watch pong_dqn.npz     # watch the trained agent play in a window
    python src/train_dqn.py --eval  pong_dqn.npz     # headless: 20 games, print win rate

Network: 7 -> 64 -> 64 -> 3 (ReLU), Huber loss, Adam, replay buffer, target network, Double-DQN targets.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pong_env import PongEnv  # noqa: E402


# ============================================================================ network
class QNet:
    """Small fully-connected Q-network with hand-written backprop and Adam."""

    def __init__(self, sizes=(7, 64, 64, 3), rng=None, dtype=np.float64, init="he"):
        rng = rng or np.random.default_rng(0)
        self.sizes = tuple(sizes)
        self.p = {}
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]), 1):
            if init == "torch":      # PyTorch nn.Linear default: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for W and b
                lim = 1.0 / np.sqrt(a)
                self.p[f"W{i}"] = rng.uniform(-lim, lim, (a, b)).astype(dtype)
                self.p[f"b{i}"] = rng.uniform(-lim, lim, b).astype(dtype)
            else:                    # He init
                self.p[f"W{i}"] = rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(dtype)
                self.p[f"b{i}"] = np.zeros(b, dtype=dtype)
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x, cache=False):
        n = len(self.sizes) - 1
        acts, pre = [x], []
        h = x
        for i in range(1, n + 1):
            z = h @ self.p[f"W{i}"] + self.p[f"b{i}"]
            pre.append(z)
            h = np.maximum(z, 0) if i < n else z
            acts.append(h)
        return (h, acts, pre) if cache else h

    def grads(self, x, dq):
        """Gradients of sum(dq * q) w.r.t. all parameters (dq = dLoss/dQ)."""
        q, acts, pre = self.forward(x, cache=True)
        n = len(self.sizes) - 1
        g, d = {}, dq
        for i in range(n, 0, -1):
            g[f"W{i}"] = acts[i - 1].T @ d
            g[f"b{i}"] = d.sum(0)
            if i > 1:
                d = (d @ self.p[f"W{i}"].T) * (pre[i - 2] > 0)
        return g

    def train_batch(self, x, actions, targets, lr=1e-3, clip=10.0, l2=0.0):
        B = len(x)
        q = self.forward(x)
        td = q[np.arange(B), actions] - targets
        dq = np.zeros_like(q)
        dq[np.arange(B), actions] = np.clip(td, -1.0, 1.0) / B           # Huber-loss gradient
        g = self.grads(x, dq)
        if l2:                                                            # L2 regularisation (weight decay)
            for k in g:
                g[k] = g[k] + l2 * self.p[k]
        norm = np.sqrt(sum((v ** 2).sum() for v in g.values()))
        scale = min(1.0, clip / (norm + 1e-8))
        self.t += 1
        for k in self.p:                                                  # Adam
            gk = g[k] * scale
            self.m[k] = 0.9 * self.m[k] + 0.1 * gk
            self.v[k] = 0.999 * self.v[k] + 0.001 * gk * gk
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        return float(np.mean(np.where(np.abs(td) <= 1, 0.5 * td ** 2, np.abs(td) - 0.5)))

    def sq_norm(self):
        """Squared L2 norm of all parameters (the quantity plotted in the paper)."""
        return float(sum((v.astype(np.float64) ** 2).sum() for v in self.p.values()))

    def copy_from(self, other):
        for k in self.p:
            self.p[k] = other.p[k].copy()

    def act(self, obs):
        return int(np.argmax(self.forward(obs[None, :])[0]))

    def save(self, path):
        np.savez(path, sizes=np.array(self.sizes), **self.p)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(tuple(int(s) for s in d["sizes"]))
        for k in net.p:
            net.p[k] = d[k]
        return net


# ============================================================================ replay buffer
class Replay:
    def __init__(self, capacity, obs_dim, rng):
        self.cap, self.n, self.i, self.rng = capacity, 0, 0, rng
        self.s = np.zeros((capacity, obs_dim))
        self.s2 = np.zeros((capacity, obs_dim))
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity)
        self.term = np.zeros(capacity)

    def add(self, s, a, r, s2, term):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.term[i] = s, a, r, s2, term
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch):
        idx = self.rng.integers(0, self.n, batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.term[idx]


# ============================================================================ helpers
def play_games(net, env, games, eps=0.0, rng=None, render=False, fps=30):
    """Run full games with an (almost) greedy policy. Returns list of (agent_pts, ai_pts, won)."""
    results = []
    for _ in range(games):
        obs, done, info = env.reset(), False, {}
        while not done:
            if rng is not None and rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                a = net.act(obs)
            obs, _, done, info = env.step(a)
            if render and not env.render(fps):
                return results
        results.append((info["agent_score"], info["ai_score"], info["won"]))
        if render:
            print(f"game over: agent {info['agent_score']} - {info['ai_score']} AI  "
                  f"({'WIN' if info['won'] else 'loss'})")
            time.sleep(1.0)
    return results


def summarize(res):
    wins = sum(w for _, _, w in res)
    return wins / max(len(res), 1), np.mean([a - b for a, b, _ in res]) if res else 0.0


# ============================================================================ modes
def train(args):
    rng = np.random.default_rng(args.seed)
    env = PongEnv(args.rom, seed=args.seed, hit_reward=args.hit_reward, max_steps=args.max_steps)
    eval_env = PongEnv(args.rom, seed=args.seed + 12345, hit_reward=0.0, max_steps=args.max_steps)
    q = QNet((env.obs_dim, 64, 64, env.n_actions), rng)
    target = QNet((env.obs_dim, 64, 64, env.n_actions), rng)
    target.copy_from(q)
    buf = Replay(args.buffer, env.obs_dim, rng)

    total, best, t0 = 0, (-1.0, -99.0), time.time()
    recent = []
    csv_file = csv_w = None
    if args.csv:                                                  # one row per episode, like the paper's curves
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        csv_file = open(args.csv, "w", newline="")
        csv_w = csv.writer(csv_file)
        csv_w.writerow(["episode", "total_steps", "steps", "agent_pts", "ai_pts", "won", "paper_reward",
                        "shaped_return", "epsilon", "avg_q", "param_sq_norm", "seconds"])
    print(f"training for up to {args.episodes} episodes (Ctrl+C to stop)\n")
    try:
        for ep in range(1, args.episodes + 1):
            obs, done, ret, info = env.reset(), False, 0.0, {}
            qsum, qn, ep_steps = 0.0, 0, 0
            while not done:
                eps = max(args.eps_end, args.eps_start - total * (args.eps_start - args.eps_end) / args.eps_decay)
                qv = q.forward(obs[None, :])[0]
                qsum, qn, ep_steps = qsum + float(qv.max()), qn + 1, ep_steps + 1
                a = int(rng.integers(env.n_actions)) if rng.random() < eps else int(np.argmax(qv))
                obs2, r, done, info = env.step(a)
                buf.add(obs, a, r, obs2, float(done and not info["truncated"]))
                obs, ret, total = obs2, ret + r, total + 1

                if total >= args.learn_start and total % args.train_every == 0:
                    s, ac, rw, s2, tm = buf.sample(args.batch)
                    best_a = np.argmax(q.forward(s2), axis=1)                       # Double DQN
                    y = rw + args.gamma * (1 - tm) * target.forward(s2)[np.arange(args.batch), best_a]
                    q.train_batch(s, ac, y, lr=args.lr)
                if total % args.target_every == 0:
                    target.copy_from(q)

            recent.append((info["agent_score"], info["ai_score"], info["won"], ret))
            recent = recent[-20:]
            if csv_w:
                ag, ai = info["agent_score"], info["ai_score"]
                lost = ai >= 10 and not info["won"]
                paper = ag - 2 * ai + (20 if info["won"] else -20 if lost else 0)
                csv_w.writerow([ep, total, ep_steps, ag, ai, int(info["won"]), paper, round(ret, 3), round(eps, 4),
                                round(qsum / max(qn, 1), 5), round(sum(float((v ** 2).sum()) for v in q.p.values()), 4),
                                round(time.time() - t0, 1)])
                csv_file.flush()
            if ep % args.log_every == 0:
                wr = np.mean([w for _, _, w, _ in recent])
                print(f"ep {ep:4d} | steps {total:7d} | eps {eps:.2f} | last20: agent {np.mean([a for a, _, _, _ in recent]):4.1f}"
                      f" - AI {np.mean([b for _, b, _, _ in recent]):4.1f} | wins {wr:4.0%} | "
                      f"{total / (time.time() - t0):5.0f} steps/s", flush=True)

            if ep % args.eval_every == 0:
                res = play_games(q, eval_env, args.eval_games)
                wr, diff = summarize(res)
                tag = ""
                if (wr, diff) > best:
                    best = (wr, diff)
                    q.save(args.save)
                    tag = "  <- new best, saved"
                print(f"   EVAL greedy: {args.eval_games} games, win rate {wr:.0%}, avg score diff {diff:+.1f}{tag}", flush=True)
    except KeyboardInterrupt:
        print("\nstopped by user")
    if csv_file:
        csv_file.close()
    q.save(args.save.replace(".npz", "_last.npz"))
    print(f"\nbest eval: win rate {best[0]:.0%}, diff {best[1]:+.1f}  -> {args.save}\nlast model -> "
          f"{args.save.replace('.npz', '_last.npz')}")


def evaluate(args):
    net = QNet.load(args.eval)
    env = PongEnv(args.rom, seed=args.seed, hit_reward=0.0, max_steps=args.max_steps)
    res = play_games(net, env, args.games)
    wr, diff = summarize(res)
    print(f"{len(res)} games: win rate {wr:.0%}, avg score {np.mean([a for a, _, _ in res]):.1f} - "
          f"{np.mean([b for _, b, _ in res]):.1f}  (agent - AI)")


def watch(args):
    net = QNet.load(args.watch)
    env = PongEnv(args.rom, seed=args.seed, hit_reward=0.0, max_steps=args.max_steps)
    print("watching the agent - close the window or press ESC to quit")
    while True:
        res = play_games(net, env, 1, render=True, fps=args.fps)
        if not res:
            break
    env.close()


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom", default=None, help="ROM path (default: ../roms/pong_dql.ch8)")
    ap.add_argument("--watch", metavar="MODEL", help="play a saved model in a window")
    ap.add_argument("--eval", metavar="MODEL", help="evaluate a saved model headless")
    ap.add_argument("--games", type=int, default=20, help="games for --eval")
    ap.add_argument("--fps", type=int, default=30, help="window speed for --watch (30 = real time)")
    ap.add_argument("--save", default="pong_dqn.npz")
    ap.add_argument("--csv", default=None, help="write one row per episode here (reward, avg Q, |params|^2 ...)")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--max-steps", type=int, default=6000, help="episode length cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hit-reward", type=float, default=0.1, help="bonus for returning the ball (0 = off)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--buffer", type=int, default=100_000)
    ap.add_argument("--learn-start", type=int, default=2000)
    ap.add_argument("--train-every", type=int, default=2)
    ap.add_argument("--target-every", type=int, default=1000)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay", type=int, default=60_000, help="steps to anneal epsilon")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-games", type=int, default=3)
    return ap.parse_args()


if __name__ == "__main__":
    a = parse()
    if a.watch:
        watch(a)
    elif a.eval:
        evaluate(a)
    else:
        train(a)
