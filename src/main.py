import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chip8 import Chip8
from display import Display
from keyboard import Keyboard  # pyright: ignore[reportMissingModuleSource, reportAttributeAccessIssue]

TIMER_HZ = 60          # CHIP-8 timers (and our render rate) run at 60 Hz
MAX_STALE_TICKS = 6    # sync fallback: ROMs that never poll the delay timer still get drawn


def parse_args():
    ap = argparse.ArgumentParser(description="CHIP-8 emulator")
    ap.add_argument("rom", help="path to a .ch8 file, e.g. roms/pong_dql.ch8")
    ap.add_argument("--cycles", type=int, default=150,
                    help="CPU cycles per 60 Hz tick (default 150 = 9 kHz; pong_dql needs ~90 to finish a step inside one tick)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (reproducible serves)")
    ap.add_argument("--no-sync", action="store_true",
                    help="present a frame every tick, even mid-draw (old behaviour, flickers)")
    return ap.parse_args()


def main():
    args = parse_args()

    chip8 = Chip8(seed=args.seed)
    chip8.load_rom(args.rom)
    display = Display()
    keyboard = Keyboard(chip8)

    print("=" * 50)
    print("CHIP-8 Emulator")
    print("  1 = up | Q = down | ESC = quit")
    print("=" * 50)

    stale = 0
    announced = False
    while keyboard.is_running():
        if not keyboard.handle_events():
            break

        # ---- exactly one 60 Hz tick per loop pass (fixed timestep => deterministic) ----
        if not chip8.halted:                 # ROM finished (VICTORY screen): stop burning CPU
            for _ in range(args.cycles):
                chip8.cycle()
        chip8.tick_timers()
        keyboard.tick()   # giữ phím tối thiểu 2 tick để ROM chắc chắn đọc được

        # ---- present only complete pictures ----
        # chip8.frame_ready is True when the ROM is idle-polling its delay timer, i.e. every
        # erase/redraw of this game step has finished. Presenting at an arbitrary cycle is
        # what made the paddles flicker.
        stale += 1
        if chip8.draw_flag and (args.no_sync or chip8.frame_ready or stale >= MAX_STALE_TICKS):
            display.render(chip8)
            chip8.draw_flag = False
            stale = 0

        if chip8.halted and not announced:
            announced = True
            left, right = chip8.V[0xE], chip8.V[3]          # pong_dql.ch8 score registers
            print(f"\nVICTORY! {'Left (you)' if left > right else 'Right (AI)'} wins  {left} - {right}")
            print("Press ESC or close the window to quit.")

        display.tick(TIMER_HZ)

    print("\nExited.")
    display.close()


if __name__ == "__main__":
    main()
