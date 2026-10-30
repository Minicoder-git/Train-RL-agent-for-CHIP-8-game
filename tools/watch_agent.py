"""
Watch any agent play in a window (real time = 30 game steps per second).

    py tools/watch_agent.py --agent dqn:models/pong_dqn.npz
    py tools/watch_agent.py --agent track --games 3
    py tools/watch_agent.py --agent smart --fps 10          # slow motion
    py tools/watch_agent.py --agent vi:models/pong_vi.npz --seed 20000   # a game from the test set

Game number i uses seed (--seed + i), so you can replay exactly the game that the benchmark CSV
reports (look up its 'seed' column). ESC or closing the window quits.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.join(HERE, "src"), os.path.join(HERE, "..", "src")):
    if os.path.isdir(cand):
        sys.path.insert(0, os.path.abspath(cand))
        break
else:
    sys.exit("cannot find the src/ folder next to this script")

from agents import make_agent  # noqa: E402
from pong_env import PongEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="track")
    ap.add_argument("--games", type=int, default=0, help="number of games (0 = until you close the window)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0, help="seed of the first game")
    ap.add_argument("--rom", default=None)
    a = ap.parse_args()

    agent = make_agent(a.agent)
    print(f"watching '{agent.name}' - close the window or press ESC to quit")
    i = 0
    env = None
    while a.games == 0 or i < a.games:
        seed = a.seed + i
        env = PongEnv(a.rom, seed=seed, hit_reward=0.0)
        obs, done, info = env.reset(), False, {}
        agent.reset(seed)
        while not done:
            obs, _, done, info = env.step(agent.act(obs))
            if not env.render(a.fps):
                env.close()
                return
        i += 1
        print(f"game {i} (seed {seed}): agent {info['agent_score']} - {info['ai_score']} AI  "
              f"{'WIN' if info['won'] else 'loss'}")
        time.sleep(1.0)
    if env:
        env.close()


if __name__ == "__main__":
    main()
