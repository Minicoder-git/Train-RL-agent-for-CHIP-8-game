"""
Compare TWO frozen agents on the same games (paired comparison).

    py tools/compare_agents.py --a dqn:models/pong_dqn.npz --b track
    py tools/compare_agents.py --a dqn:models/pong_dqn.npz --b vi:models/pong_vi.npz --games 100

Why paired: both agents play the same seeds, so differences between them are not caused by one of
them getting easier serves. The test then only looks at games where the two disagree.
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

from agents import make_agent  # noqa: E402  # type: ignore[reportMissingImports]
from evaluate import evaluate, format_table, paired_compare, seeds, summarize, write_csv  # noqa: E402  # type: ignore[reportMissingImports]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="agent A, e.g. dqn:models/pong_dqn.npz")
    ap.add_argument("--b", required=True, help="agent B, e.g. track")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--rom", default=None)
    ap.add_argument("--max-steps", type=int, default=30_000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(SRC), "results"))
    a = ap.parse_args()

    seed_list = seeds(a.split, a.games)
    ag_a, ag_b = make_agent(a.a), make_agent(a.b)
    print(f"{len(seed_list)} games each, split '{a.split}' (same seeds for both agents)\n")
    rows_a = evaluate(ag_a, seed_list, a.rom, a.max_steps)
    rows_b = evaluate(ag_b, seed_list, a.rom, a.max_steps)
    sa, sb = summarize(rows_a), summarize(rows_b)
    print(format_table([(f"A  {ag_a.name}", sa), (f"B  {ag_b.name}", sb)]))

    c = paired_compare(rows_a, rows_b)
    print(f"""
Per-seed outcomes ({c['games']} seeds)
                       B wins   B does not win
  A wins              {c['both_win']:6d}   {c['only_a']:6d}
  A does not win      {c['only_b']:6d}   {c['neither']:6d}

Decisive seeds (only one agent wins): A {c['only_a']}, B {c['only_b']}  -> exact sign test p = {c['p_value']:.4f}
Score difference gap (A - B), mean per game: {c['mean_diff_gap']:+.2f}  95% CI [{c['gap_ci_low']:+.2f}, {c['gap_ci_high']:+.2f}]
""")
    if c["only_a"] + c["only_b"] == 0:
        print("Verdict: the agents win the same games - win rate cannot separate them. "
              "Look at the score-difference gap above, or test against a harder opponent.")
    elif c["p_value"] < 0.05:
        better = "A" if c["only_a"] > c["only_b"] else "B"
        print(f"Verdict: {better} is better, and the difference is unlikely to be luck (p < 0.05).")
    else:
        print("Verdict: no reliable difference (p >= 0.05). Play more games or compare on another metric.")

    write_csv([{"agent": "A:" + ag_a.name, **r} for r in rows_a] + [{"agent": "B:" + ag_b.name, **r} for r in rows_b],
              os.path.join(a.out, "compare_games.csv"))
    print(f"\nPer-game CSV: {os.path.join(a.out, 'compare_games.csv')}")


if __name__ == "__main__":
    main()
