"""
Headless checks for roms/pong_dql.ch8 (no pygame / window needed).

    python tools/test_pong.py

1. FLICKER    every frame the main loop would present equals the picture implied by the
              registers (2 paddles x 6 px, 1 ball px, score digits) - never half-drawn.
2. TUNNELLING ball stays in x 3..62, paddles stay in y 0..26, and every contact with a paddle
              plane is a HIT exactly when the ball row is inside the paddle's rows.
3. PACING     one game step == STEP_TICKS ticks, independent of CPU speed, and the step's
              work fits inside one 100-cycle tick.
4. DETERMINISM same seed + same inputs -> identical state trace.
5. FAIRNESS   a ball-tracking agent never misses (so a learning agent is only ever beaten
              by its own policy, not by the physics).
6. GAME OVER  first to 10 ends the game: scores never exceed 10, "VICTORY" is drawn on the
              winner's half, and the ROM halts for good (keys are ignored).
"""
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import build_pong                      # noqa: E402
from chip8 import Chip8                # noqa: E402  # pyright: ignore[reportMissingImports]  # pyright: ignore[reportMissingImports]

CYCLES = 100
ROM, L = build_pong.build()
STEP_TICKS = build_pong.make_consts()["STEP_TICKS"]


# independent copy of the 3x5 glyphs, written as picture rows
GLYPHS = {
    "V": ["#.#", "#.#", "#.#", "#.#", ".#."],
    "I": ["###", ".#.", ".#.", ".#.", "###"],
    "C": ["###", "#..", "#..", "#..", "###"],
    "T": ["###", ".#.", ".#.", ".#.", ".#."],
    "O": ["###", "#.#", "#.#", "#.#", "###"],
    "R": ["##.", "#.#", "##.", "#.#", "#.#"],
    "Y": ["#.#", "#.#", ".#.", ".#.", ".#."],
}
GLYPHS = {k: [sum(0x80 >> i for i, ch in enumerate(r) if ch == "#") for r in v] for k, v in GLYPHS.items()}


def new_game(seed):
    c = Chip8(seed=seed)
    for i, b in enumerate(ROM):
        c.memory[0x200 + i] = b
    return c


def expected_frame(c):
    """Picture implied by the game state (independent of how the ROM drew it)."""
    f = [[0] * 64 for _ in range(32)]

    def xor(x, y):
        f[y % 32][x % 64] ^= 1

    for r in range(6):
        xor(c.V[0xA], c.V[0xB] + r)
        xor(c.V[0xC], c.V[0xD] + r)
    ended = c.pc in (L["halt"],)
    if not ended:                                              # ball is erased on game over
        xor(c.V[6], c.V[7])
    if ended:
        x0 = 5 if c.V[0xE] > c.V[3] else 35                    # winner's half
        for i, ch in enumerate("VICTORY"):
            for r, row in enumerate(GLYPHS[ch]):
                for col in range(3):
                    if row & (0x80 >> col):
                        xor(x0 + 4 * i + col, 13 + r)
    for score, x0 in ((c.V[0xE], 12), (c.V[3], 42)):          # agent, AI
        for d, dx in ((score // 10, 0), (score % 10, 5)):
            if dx == 0 and d == 0:
                continue                                      # blank leading zero
            for r in range(5):
                row = c.memory[d * 5 + r]
                for col in range(8):
                    if row & (0x80 >> col):
                        xor(x0 + dx + col, r)
    return f


def run(c, ticks, policy, on_cycle=None, on_present=None, on_tick=None):
    """Exactly what src/main.py does, minus pygame."""
    for t in range(ticks):
        c.keypad[1], c.keypad[4] = policy(c, t)
        for _ in range(CYCLES):
            if on_cycle:
                on_cycle(c)
            c.cycle()
        c.tick_timers()
        if c.draw_flag and c.frame_ready:
            if on_present:
                on_present(c)
            c.draw_flag = False
        if on_tick:
            on_tick(c, t)


def random_policy(seed):
    rng = random.Random(seed)
    state = [0, 0, 0]                              # up, down, ticks left

    def policy(c, t):
        if state[2] <= 0:
            state[0], state[1] = rng.choice([(0, 0), (1, 0), (0, 1), (1, 0), (0, 1)])
            state[2] = rng.randint(2, 30)
        state[2] -= 1
        return state[0], state[1]
    return policy


def tracker_policy(c, t):
    """Perfect-ish agent: keep paddle centre (y+2.5) on the ball."""
    centre = c.V[0xB] + 2.5
    if c.V[7] < centre - 1.5:
        return 1, 0
    if c.V[7] > centre + 1.5:
        return 0, 1
    return 0, 0


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return ok


def main():
    ok = True
    print(f"ROM: {len(ROM)} bytes, {60 / STEP_TICKS:.0f} steps/s, {CYCLES} cycles/tick\n")

    # ---------------------------------------------------------------- 1 + 2
    print("1/2  flicker + tunnelling fuzz (random inputs, 30 games, random inputs, up to 3 min each)")
    presented = bad_frames = 0
    out_of_range = 0
    contacts = hits = misses = wrong = 0
    flat = {"n": 0, "bad": 0, "dirs": set()}
    pend = {}

    def on_present(c):
        nonlocal presented, bad_frames
        presented += 1
        if c.display != expected_frame(c):
            bad_frames += 1

    def on_cycle(c):
        nonlocal out_of_range, contacts, hits, misses, wrong, flat
        pc = c.pc
        if pc == L["step_wait"] or pc == L["loop"]:
            if not (3 <= c.V[6] <= 62 and 0 <= c.V[7] <= 31 and c.V[0xB] <= 26 and c.V[0xD] <= 26):
                out_of_range += 1
        if pc in (L["left_contact"], L["right_contact"]):
            left = pc == L["left_contact"]
            py = c.V[0xB] if left else c.V[0xD]
            # touching = the row the ball is drawn on OR the row it moves to is on the paddle
            row = c.V[5] if py <= c.V[5] <= py + 5 else c.V[7]     # row the ROM will use
            pend.update(left=left, should_hit=py <= c.V[5] <= py + 5 or py <= c.V[7] <= py + 5,
                        vy0=c.V[9], off=row - py)
            contacts += 1
        elif pc in (L["bounce"], L["left_miss"], L["right_miss"]) and pend:
            did_hit = pc == L["bounce"]
            hits += did_hit
            misses += not did_hit
            if did_hit != pend["should_hit"]:
                wrong += 1
            if did_hit and pend["vy0"] == 0 and pend["off"] in (2, 3):      # flat ball, middle hit
                flat["n"] += 1
                flat["dirs"].add(c.V[9])
                flat["bad"] += c.V[9] not in (1, 0xFF)
            pend.clear()

    for seed in range(30):
        c = new_game(seed)
        run(c, 60 * 180, random_policy(seed), on_cycle=on_cycle, on_present=on_present)
    ok &= check("presented frames match game state", bad_frames == 0,
                f"({bad_frames} bad / {presented} frames)")
    ok &= check("ball x in 3..62, y in 0..31, paddles y in 0..26", out_of_range == 0,
                f"({out_of_range} violations)")
    ok &= check("every paddle contact resolved correctly", wrong == 0 and contacts > 100,
                f"({contacts} contacts: {hits} hits, {misses} misses, {wrong} wrong)")

    ok &= check("flat ball hitting the middle is always kicked up or down (both ways seen)",
                flat["n"] > 10 and flat["bad"] == 0 and flat["dirs"] == {1, 0xFF},
                f"({flat['n']} flat middle hits, {flat['bad']} stayed flat, directions {sorted(flat['dirs'])})")

    # ---------------------------------------------------------------- 3
    print("\n3    pacing")
    for cycles_per_tick in (100, 400):               # slow and fast virtual CPU
        c = new_game(1)
        starts, last_ticks = [], []
        tick = [0]
        cur = []

        def oc(c):
            if c.pc == L["loop"]:
                starts.append(tick[0])
            if c.pc == L["step_wait"]:
                cur.append(1)

        for t in range(60 * 30):
            c.keypad[1] = c.keypad[4] = 0
            for _ in range(cycles_per_tick):
                oc(c)
                c.cycle()
            c.tick_timers()
            tick[0] += 1
        gaps = [b - a for a, b in zip(starts, starts[1:]) if b - a < 20]   # ignore serve pauses
        ok &= check(f"step period at {cycles_per_tick} cycles/tick", set(gaps) == {STEP_TICKS},
                    f"(gaps seen: {sorted(set(gaps))}, expected {{{STEP_TICKS}}})")

    c = new_game(3)
    worst = [0]
    count = [0]
    armed = [False]

    def oc2(c):
        if c.pc == L["loop"]:
            count[0], armed[0] = 0, True
        count[0] += 1
        if c.pc == L["step_wait"] and armed[0]:      # first arrival only (not the idle spin)
            worst[0] = max(worst[0], count[0])
            armed[0] = False

    run(c, 60 * 120, random_policy(3), on_cycle=oc2)
    ok &= check("longest step fits in one tick", worst[0] < CYCLES, f"({worst[0]} cycles of {CYCLES})")

    # ---------------------------------------------------------------- 4
    print("\n4    determinism")

    def trace(seed):
        c = new_game(seed)
        out = []
        run(c, 60 * 30, random_policy(99), on_tick=lambda c, t: out.append(tuple(c.V) + (c.pc,)))
        return out
    ok &= check("same seed -> identical trace", trace(7) == trace(7))
    ok &= check("different seed -> different trace", trace(7) != trace(8))

    # ---------------------------------------------------------------- 5
    print("\n5    fairness / learnability")
    c = new_game(11)
    run(c, 60 * 600, tracker_policy)
    ok &= check("tracking agent is never beaten by the physics", c.V[3] == 0,
                f"(agent {c.V[0xE]} pts, AI {c.V[3]} pts after 10 min of game time)")

    # ---------------------------------------------------------------- 6
    print("\n6    game over at 10 points -> VICTORY + halt")
    games = bad_end = not_halted = not_frozen = bad_pic = never_ended = 0
    winners = {"agent": 0, "ai": 0}
    for seed in range(40):
        c = new_game(100 + seed)
        pol = random_policy(seed)
        mx = [0]
        run(c, 60 * 60 * 6, pol, on_cycle=lambda c: mx.__setitem__(0, max(mx[0], c.V[0xE], c.V[3])),
            on_tick=None)
        if not c.halted:
            never_ended += 1
            continue
        games += 1
        a_, b_ = c.V[0xE], c.V[3]
        if sorted((a_, b_))[1] != 10 or min(a_, b_) >= 10 or mx[0] != 10:
            bad_end += 1
        winners["agent" if a_ == 10 else "ai"] += 1
        c.draw_flag = False
        snap = (tuple(c.V), c.pc, [r[:] for r in c.display])
        run(c, 60 * 10, random_policy(seed + 1000))                    # keys mashed for 10 s
        if (tuple(c.V), c.pc, c.display) != snap:
            not_frozen += 1
        # the picture that was presented at the end shows the VICTORY text on the right side
        if c.display != expected_frame(c):
            bad_pic += 1
    ok &= check("every game ends with exactly one side on 10", bad_end == 0 and never_ended == 0 and games == 40,
                f"({games}/40 finished; agent won {winners['agent']}, AI won {winners['ai']}; {never_ended} never ended)")
    ok &= check("keys are ignored after the end (state + picture frozen)", not_frozen == 0, f"({not_frozen} changed)")
    ok &= check("final picture = paddles + score + VICTORY on winner's half", bad_pic == 0, f"({bad_pic} mismatches)")

    print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
