"""
Gym-style environment around roms/pong_dql.ch8 for reinforcement-learning agents.

    env = PongEnv()                      # headless, no window
    obs = env.reset()
    obs, reward, done, info = env.step(action)

Actions   0 = stay, 1 = up (CHIP-8 key 1), 2 = down (CHIP-8 key 4)

Observation (7 floats, read straight from the ROM's registers - no pixels):
    0  ball x / 63                 4  agent paddle centre y / 31
    1  ball y / 31                 5  (ball y - agent paddle centre) / 31
    2  ball vx sign  (-1 / +1)     6  AI paddle centre y / 31
    3  ball vy       (-1 / 0 / +1)

Reward    +1 when the agent scores, -1 when the AI scores,
          +hit_reward when the agent returns the ball (helps early learning; set 0 to disable)

Paper mode (Achenjang, DeMichele, Rogers, "Playing CHIP-8 Games with Reinforcement Learning"):
    PongEnv(obs_mode="pixels", reward_mode="paper", ticks_per_step=6)
    obs    = the previous and the current 32x64 screen, flattened (4096 values, 0/1)
    reward = +1 per point scored, -2 per point conceded, +20 for winning / -20 for losing the game

done      True when someone reaches 10 points (the ROM halts) or max_steps is hit.
          info["truncated"] is True in the max_steps case.

One env step == one game step of the ROM (STEP_TICKS = 2 ticks of 60 Hz = 1/30 s of game time).
The pause after each point is fast-forwarded, and idle waiting inside the ROM is skipped, so the
game plays exactly like `main.py` but runs thousands of steps per second.
If you rebuild the ROM with --step-ticks N, pass ticks_per_step=N here.
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chip8 import Chip8  # noqa: E402

KEY_UP, KEY_DOWN = 0x1, 0x4
DEFAULT_ROM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "roms", "pong_dql.ch8")


class PongEnv:
    n_actions = 3

    def __init__(self, rom_path=None, seed=None, ticks_per_step=2, cycles_per_tick=150,
                 hit_reward=0.1, max_steps=20000, obs_mode="registers", reward_mode="shaped"):
        assert obs_mode in ("registers", "pixels") and reward_mode in ("shaped", "paper")
        self.obs_mode, self.reward_mode = obs_mode, reward_mode
        self.obs_dim = 7 if obs_mode == "registers" else 2 * 32 * 64
        self._prev_screen = None
        self.rom_path = rom_path or DEFAULT_ROM
        self.ticks_per_step = ticks_per_step
        self.cycles_per_tick = cycles_per_tick
        self.hit_reward = hit_reward
        self.max_steps = max_steps
        self._seeds = random.Random(seed)       # one fresh, reproducible ROM seed per episode
        self._display = None
        self.chip8 = None
        self.steps = 0

    # ------------------------------------------------------------------ core API
    def reset(self):
        self.chip8 = Chip8(seed=self._seeds.randrange(2 ** 31))
        self.chip8.load_rom(self.rom_path)
        self.steps = 0
        self._prev_screen = None
        # run the intro + first serve pause (ROM sets the delay timer at least once, then counts down)
        while not self.chip8.frame_ended or self.chip8.delay_timer > 0:
            self._tick()
        return self._obs()

    def step(self, action):
        c = self.chip8
        c.keypad[KEY_UP] = int(action == 1)
        c.keypad[KEY_DOWN] = int(action == 2)
        agent0, ai0 = c.V[0xE], c.V[3]
        heading_to_agent = c.V[8] >= 128

        for _ in range(self.ticks_per_step):
            self._tick()
        while c.delay_timer > 0 and not c.halted:          # fast-forward the pause after a point
            self._tick()

        if self.reward_mode == "paper":
            reward = float(c.V[0xE] - agent0) - 2.0 * float(c.V[3] - ai0)
            if c.halted:
                reward += 20.0 if c.V[0xE] > c.V[3] else -20.0
        else:
            reward = float(c.V[0xE] - agent0) - float(c.V[3] - ai0)
            if reward == 0 and heading_to_agent and c.V[8] < 128:   # ball turned around -> agent hit it
                reward += self.hit_reward

        self.steps += 1
        truncated = self.steps >= self.max_steps and not c.halted
        done = c.halted or truncated
        if done:
            c.keypad[KEY_UP] = c.keypad[KEY_DOWN] = 0
        info = {"agent_score": c.V[0xE], "ai_score": c.V[3], "truncated": truncated,
                "won": bool(c.halted and c.V[0xE] > c.V[3])}
        return self._obs(), reward, done, info

    # ------------------------------------------------------------------ helpers
    def _tick(self):
        """One 60 Hz tick. Stops early when the ROM is just spinning on its delay timer."""
        c = self.chip8
        mem = c.memory
        for _ in range(self.cycles_per_tick):
            op = (mem[c.pc] << 8) | mem[c.pc + 1]
            nxt = mem[c.pc + 2] & 0xF0                      # high nibble of the following opcode
            c.cycle()
            if c.halted:
                break
            # spin-wait pattern  "LD Vx,DT ; SE Vx,0 ; JP back"  with DT > 0: nothing can change until the
            # next tick, so skip the rest of this one (the halt loop "LD V0,DT ; JP self" is NOT skipped)
            if (op & 0xF0FF) == 0xF007 and nxt == 0x30 and c.V[(op >> 8) & 0xF] != 0:
                break
        c.tick_timers()

    def _obs(self):
        if self.obs_mode == "pixels":
            cur = np.array(self.chip8.display, dtype=np.uint8).reshape(-1)       # 32 x 64 booleans
            prev = cur if self._prev_screen is None else self._prev_screen
            self._prev_screen = cur
            return np.concatenate([prev, cur])
        V = self.chip8.V
        bx, by = V[6], V[7]
        vx = -1.0 if V[8] >= 128 else 1.0
        vy = float(V[9] - 256 if V[9] >= 128 else V[9])
        me, ai = V[0xB] + 2.5, V[0xD] + 2.5
        return np.array([bx / 63, by / 31, vx, vy, me / 31, (by - me) / 31, ai / 31], dtype=np.float64)

    # ------------------------------------------------------------------ optional window
    def render(self, fps=30):
        """Draw the current frame in a pygame window. Returns False if the user closed it / pressed ESC."""
        try:
            import pygame
        except ImportError:
            raise SystemExit("Viewing needs pygame:  pip install pygame   (training/evaluation do not)")
        if self._display is None:
            from display import Display
            self._display = Display()
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return False
        self._display.render(self.chip8)
        self._display.tick(fps)
        return True

    def close(self):
        if self._display is not None:
            self._display.close()
            self._display = None
