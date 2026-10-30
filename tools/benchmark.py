"""
Run several FROZEN agents on exactly the same games and print one comparison table.

    py tools/benchmark.py                                   # the built-in baselines
    py tools/benchmark.py --agents random track smart dqn:models/pong_dqn.npz vi:models/pong_vi.npz
    py tools/benchmark.py --split val --games 50

Outputs (folder results/ next to src/):
    benchmark_games.csv     one row per game per agent (seed, points, win, steps, paper reward ...)
    benchmark_summary.csv   one row per agent
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.join(HERE, "src"), os.path.join(HERE, "..", "src")):
    if os.path.isdir(cand):
        SRC = os.path.abspath(cand)
        sys.path.insert(0, SRC)
        break
else:
    sys.exit("cannot find the src/ folder next to this script")

from agents import make_agent  # pyright: ignore[reportMissingImports] # noqa: E402
from evaluate import evaluate, format_table, seeds, summarize, write_csv  # pyright: ignore[reportMissingImports] # noqa: E402

DEFAULT_AGENTS = ["random", "idle", "track", "top", "bottom", "smart"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", nargs="+", default=DEFAULT_AGENTS)
    ap.add_argument("--split", choices=["val", "test"], default="test",
                    help="which fixed set of game seeds to play (default test)")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--rom", default=None)
    ap.add_argument("--max-steps", type=int, default=30_000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(SRC), "results"))
    a = ap.parse_args()

    seed_list = seeds(a.split, a.games)
    print(f"{len(seed_list)} games per agent, split '{a.split}', seeds {seed_list[0]}..{seed_list[-1]}\n")

    all_rows, summaries = [], []
    for spec in a.agents:
        agent = make_agent(spec)
        rows = evaluate(agent, seed_list, a.rom, a.max_steps,
                        progress=lambda i, n, s=spec: print(f"\r  {s:30s} {i}/{n}", end="", flush=True))
        print()
        s = summarize(rows)
        summaries.append((agent.name, s))
        all_rows += [{"agent": agent.name, **r} for r in rows]

    print("\n" + format_table(summaries))
    write_csv(all_rows, os.path.join(a.out, "benchmark_games.csv"))
    write_csv([{"agent": n, **s} for n, s in summaries], os.path.join(a.out, "benchmark_summary.csv"))
    print(f"\nCSV written to {a.out}")


if __name__ == "__main__":
    main()
