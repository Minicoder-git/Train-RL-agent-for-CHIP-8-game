"""
Build roms/pong_dql.ch8 - a flicker-free, tunnel-free, fixed-speed Pong for CHIP-8.

    python tools/build_pong.py                       # default: 30 game steps / s
    python tools/build_pong.py --step-ticks 4        # 15 steps / s (slower)
    python tools/build_pong.py --ball-speed 2        # ball moves 2 px / step
    python tools/build_pong.py --ai-rest 8           # stronger AI (rests 1 step in 8; default 4, old 2)

Game speed is decided by the ROM, not by the emulator's CPU speed:
    steps per second = 60 / STEP_TICKS
    one step = ball moves BALL_SPEED px, agent paddle moves up to PADDLE_SPEED px

Register map (read these directly to build a DQL state vector, no pixels needed):
    V6,V7   ball x,y            V8,V9   ball vx,vy (signed 8-bit)
    VA,VB   left  paddle x,y    (player / agent)   keys: CHIP-8 1 = up, 4 = down
    VC,VD   right paddle x,y    (built-in AI)
    VE      left  player's (agent) points      V3   right player's (AI) points
    First to WIN_SCORE (10) wins: "VICTORY" is drawn on the winner's half of the screen and the
    ROM halts for good (self-jump at label halt -> chip8.halted == True).
"""
import argparse
import pathlib
import re
import sys


# --------------------------------------------------------------------------- tiny assembler
def _bytes_of(line, consts):
    return [int(eval(t, {}, consts)) & 0xFF for t in line[len(".byte"):].split(",")]


def _reg(tok):
    tok = tok.strip()
    return int(tok[1:], 16) if re.fullmatch(r"V[0-9A-F]", tok) else None


def assemble(src, consts):
    labels, lines, pc = {}, [], 0x200

    # pass 1: addresses
    for raw in src.splitlines():
        line = raw.split(";")[0].strip()
        while ":" in line:
            name, line = line.split(":", 1)
            labels[name.strip()] = pc
            line = line.strip()
        if not line:
            continue
        lines.append(line)
        pc += len(_bytes_of(line, consts)) if line.startswith(".byte") else 2

    def num(tok):
        tok = tok.strip()
        if tok in labels:
            return labels[tok]
        return int(eval(tok, {}, dict(consts)))

    # pass 2: encode
    out = bytearray()
    for line in lines:
        if line.startswith(".byte"):
            out += bytes(_bytes_of(line, consts))
            continue
        mnem, _, rest = line.partition(" ")
        a = [t.strip() for t in rest.split(",")] if rest else []
        mnem = mnem.upper()
        R = lambda i: _reg(a[i])
        op = None
        if mnem == "CLS":
            op = 0x00E0
        elif mnem == "RET":
            op = 0x00EE
        elif mnem == "JP":
            op = 0x1000 | num(a[0])
        elif mnem == "CALL":
            op = 0x2000 | num(a[0])
        elif mnem in ("SE", "SNE"):
            base_reg, base_imm = (0x5000, 0x3000) if mnem == "SE" else (0x9000, 0x4000)
            if R(1) is not None:
                op = base_reg | R(0) << 8 | R(1) << 4
            else:
                op = base_imm | R(0) << 8 | (num(a[1]) & 0xFF)
        elif mnem == "LD":
            d, s = a
            if d == "I":
                op = 0xA000 | num(s)
            elif d == "DT":
                op = 0xF015 | _reg(s) << 8
            elif d == "ST":
                op = 0xF018 | _reg(s) << 8
            elif d == "F":
                op = 0xF029 | _reg(s) << 8
            elif d == "B":
                op = 0xF033 | _reg(s) << 8
            elif d == "[I]":
                op = 0xF055 | _reg(s) << 8
            elif s == "DT":
                op = 0xF007 | _reg(d) << 8
            elif s == "[I]":
                op = 0xF065 | _reg(d) << 8
            elif s == "K":
                op = 0xF00A | _reg(d) << 8
            elif _reg(s) is not None:
                op = 0x8000 | _reg(d) << 8 | _reg(s) << 4
            else:
                op = 0x6000 | _reg(d) << 8 | (num(s) & 0xFF)
        elif mnem == "ADD" and a[0] == "I":
            op = 0xF01E | _reg(a[1]) << 8
        elif mnem == "ADD":
            if R(1) is not None:
                op = 0x8004 | R(0) << 8 | R(1) << 4
            else:
                op = 0x7000 | R(0) << 8 | (num(a[1]) & 0xFF)
        elif mnem in ("OR", "AND", "XOR", "SUB", "SUBN"):
            low = {"OR": 1, "AND": 2, "XOR": 3, "SUB": 5, "SUBN": 7}[mnem]
            op = 0x8000 | R(0) << 8 | R(1) << 4 | low
        elif mnem == "RND":
            op = 0xC000 | R(0) << 8 | (num(a[1]) & 0xFF)
        elif mnem == "DRW":
            op = 0xD000 | R(0) << 8 | R(1) << 4 | (num(a[2]) & 0xF)
        elif mnem == "SKP":
            op = 0xE09E | R(0) << 8
        elif mnem == "SKNP":
            op = 0xE0A1 | R(0) << 8
        if op is None:
            sys.exit(f"cannot assemble: {line}")
        out += bytes([op >> 8, op & 0xFF])
    return bytes(out), labels


# --------------------------------------------------------------------------- game source
SRC = r"""
; ---------------------------------------------------------------- init
start:
    LD   I, aicnt
    LD   V0, 0
    LD   [I], V0                ; AI step counter = 0
    LD   VA, LEFT_X
    LD   VB, 13                 ; (32-6)/2 -> vertically centred
    LD   VC, RIGHT_X
    LD   VD, 13
    LD   VE, 0
    LD   V3, 0                  ; AI score
    CLS
    LD   I, paddle
    DRW  VA, VB, 6
    DRW  VC, VD, 6
    CALL score
    LD   V6, 3                  ; first serve: from the left, towards the AI
    LD   V8, BALL_SPEED
    CALL serve

; ---------------------------------------------------------------- one game step
loop:
    LD   V0, STEP_TICKS         ; the whole step is paced by the delay timer
    LD   DT, V0

    ; ---- left paddle (player / agent): new y is built in V1, drawn once
    LD   V1, VB
    LD   V0, 1
    SKNP V0
    CALL pad_up
    LD   V0, 4
    SKNP V0
    CALL pad_down
    SE   V1, VB                 ; only touch the screen if the paddle actually moved
    CALL move_left

    ; ---- right paddle (AI)
    CALL ai
    SE   V1, VD
    CALL move_right

    ; ---- ball: compute the next position first, draw afterwards
    LD   V4, V6
    ADD  V4, V8                 ; next x
    LD   V5, V7
    ADD  V5, V9                 ; next y
    LD   V0, V4
    LD   V1, 3
    SUB  V0, V1                 ; VF = (next x >= 3)
    SNE  VF, 0
    JP   left_contact           ; reached or crossed the left paddle plane (range test, not ==)
    LD   V0, 62
    SUB  V0, V4                 ; VF = (62 >= next x)
    SNE  VF, 0
    JP   right_contact          ; reached or crossed the right paddle plane

commit:                         ; V4,V5 = final position, V8,V9 = final velocity
    SNE  V5, 31                 ; bottom wall
    LD   V9, 0xFF
    SNE  V5, 0                  ; top wall
    LD   V9, 1
    LD   I, ball
    DRW  V6, V7, 1              ; erase old ball ...
    LD   V6, V4
    LD   V7, V5
    DRW  V6, V7, 1              ; ... and draw the new one, back to back

step_wait:                      ; ROM is idle here: picture is complete (emulator presents the frame)
    LD   V0, DT
    SE   V0, 0
    JP   step_wait
    JP   loop

; ---------------------------------------------------------------- contacts
left_contact:                   ; hit if the ball's NEXT row or its CURRENT (drawn) row touches the paddle
    LD   V0, V5
    LD   V1, VB
    CALL in_paddle
    SNE  VF, 1
    JP   left_hit
    LD   V0, V7
    LD   V1, VB
    CALL in_paddle
    SNE  VF, 0
    JP   left_miss
left_hit:
    CALL zone                   ; V0 = hit offset
    LD   V0, 6                  ; reflect x about the plane x = 3
    SUB  V0, V4
    LD   V4, V0
    JP   bounce

right_contact:
    LD   V0, V5
    LD   V1, VD
    CALL in_paddle
    SNE  VF, 1
    JP   right_hit
    LD   V0, V7
    LD   V1, VD
    CALL in_paddle
    SNE  VF, 0
    JP   right_miss
right_hit:
    CALL zone
    LD   V0, 124                ; reflect x about the plane x = 62
    SUB  V0, V4
    LD   V4, V0
bounce:
    LD   V0, 0
    SUBN V8, V0                 ; vx = -vx
    LD   V0, 4
    LD   ST, V0
    JP   commit

left_miss:                      ; AI scores; next serve goes to the loser
    LD   I, ball
    DRW  V6, V7, 1
    LD   V0, 0x20
    LD   ST, V0
    CALL score
    ADD  V3, 1                  ; AI scores
    CALL score
    SNE  V3, WIN_SCORE
    JP   ai_wins
    LD   V6, 62
    LD   V8, NEG_BALL_SPEED
    JP   new_serve

right_miss:
    LD   I, ball
    DRW  V6, V7, 1
    LD   V0, 0x20
    LD   ST, V0
    CALL score
    ADD  VE, 1                  ; agent scores
    CALL score
    SNE  VE, WIN_SCORE
    JP   agent_wins
    LD   V6, 3
    LD   V8, BALL_SPEED
new_serve:
    CALL serve
    JP   loop

; ---------------------------------------------------------------- subroutines
serve:                          ; in: V6 = x, V8 = vx. Draws the ball, then pauses
    RND  V7, 0x0F
    ADD  V7, 8
    LD   V9, 1
    RND  V0, 1
    SNE  V0, 1
    LD   V9, 0xFF
    LD   I, ball
    DRW  V6, V7, 1
    LD   V0, SERVE_TICKS
    LD   DT, V0
serve_wait:
    LD   V0, DT
    SE   V0, 0
    JP   serve_wait
    RET

in_paddle:                      ; in: V0 = ball y, V1 = paddle y. out: VF = 1 if the ball row is on the
    SUB  V0, V1                 ;     paddle (6 rows), V0 = offset 0..5
    SNE  VF, 0                  ; ball above the paddle -> VF = 0
    RET
    LD   V2, 5
    SUB  V2, V0                 ; VF = (5 >= offset)
    RET

zone:                           ; in: V0 = hit offset 0..5 -> bend vy (top: up, bottom: down, flat+middle: random)
    LD   V1, 2
    SUB  V0, V1
    SNE  VF, 0
    JP   vy_up
    SUB  V0, V1
    SNE  VF, 1
    JP   vy_down
    SNE  V9, 0                  ; middle hit: if the ball is flying flat (vy = 0) ...
    JP   vy_random              ; ... kick it randomly up or down, so a rally can't stay horizontal
    RET
vy_random:
    RND  V0, 1
    LD   V9, 1
    SNE  V0, 1
    LD   V9, 0xFF
    RET
vy_up:
    ADD  V9, 0xFF
    SNE  V9, 0xFE
    LD   V9, 0xFF
    RET
vy_down:
    ADD  V9, 1
    SNE  V9, 2
    LD   V9, 1
    RET

pad_up:                         ; paddle y is CLAMPED to 0..26, never wrapped
    LD   V0, PADDLE_SPEED
    JP   dec_y
pad_down:
    LD   V0, PADDLE_SPEED
    JP   inc_y
dec_y:                          ; V1 -= V0, floor 0
    SUB  V1, V0
    SE   VF, 1
    LD   V1, 0
    RET
inc_y:                          ; V1 += V0, ceiling 26
    ADD  V1, V0
    LD   V2, 26
    SUB  V2, V1
    SE   VF, 1
    LD   V1, 26
    RET

move_left:
    LD   I, paddle
    DRW  VA, VB, 6              ; erase
    LD   VB, V1
    DRW  VA, VB, 6              ; draw (immediately after, same frame)
    RET
move_right:
    LD   I, paddle
    DRW  VC, VD, 6
    LD   VD, V1
    DRW  VC, VD, 6
    RET

ai:                             ; out: V1 = wanted y of the right paddle (= VD when it should stay)
    LD   V1, VD
    LD   I, aicnt
    LD   V0, [I]
    ADD  V0, 1
    LD   [I], V0
    LD   V2, AI_MASK
    AND  V0, V2
    SNE  V0, AI_REST_VAL        ; AI rests on every AI_REST-th step (AI_REST_VAL=1 -> never rests)
    RET
    LD   V2, VD
    ADD  V2, 2
    LD   V4, V7
    SUB  V4, V2                 ; VF = ball y >= paddle y + 2
    SNE  VF, 0
    JP   ai_up
    ADD  V2, 2
    LD   V4, V7
    SUB  V4, V2                 ; VF = ball y >= paddle y + 4
    SNE  VF, 1
    JP   ai_down
    RET
ai_up:
    LD   V0, AI_SPEED
    JP   dec_y
ai_down:
    LD   V0, AI_SPEED
    JP   inc_y

score:                          ; draws both scores (XOR: call once to erase, once to draw)
    LD   V0, VE
    LD   V4, 12
    CALL num
    LD   V0, V3
    LD   V4, 42
num:                            ; in: V0 = value 0..99, V4 = x of tens digit. Blank leading zero
    LD   I, bcd
    LD   B, V0
    LD   V2, [I]
    LD   V5, 0
    LD   F, V1
    SE   V1, 0
    DRW  V4, V5, 5
    ADD  V4, 5
    LD   F, V2
    DRW  V4, V5, 5
    RET

agent_wins:                     ; VICTORY on the left half
    LD   V4, 5
    JP   victory
ai_wins:                        ; VICTORY on the right half
    LD   V4, 35
victory:                        ; in: V4 = x of first letter. 7 letters, 3x5 px, 4 px pitch
    LD   I, letters
    LD   V1, 7
    LD   V5, 13
    LD   V2, 5
victory_loop:
    DRW  V4, V5, 5
    ADD  I, V2
    ADD  V4, 4
    ADD  V1, 0xFF
    SE   V1, 0
    JP   victory_loop
game_end:
    LD   V0, DT                 ; tells the emulator the picture is complete
halt:
    JP   halt                   ; the game is over for good

paddle: .byte 0x80,0x80,0x80,0x80,0x80,0x80
ball:   .byte 0x80
bcd:    .byte 0,0,0
aicnt:  .byte 0
letters:                        ; V I C T O R Y  (3 px wide glyphs)
    .byte 0xA0,0xA0,0xA0,0xA0,0x40
    .byte 0xE0,0x40,0x40,0x40,0xE0
    .byte 0xE0,0x80,0x80,0x80,0xE0
    .byte 0xE0,0x40,0x40,0x40,0x40
    .byte 0xE0,0xA0,0xA0,0xA0,0xE0
    .byte 0xC0,0xA0,0xC0,0xA0,0xA0
    .byte 0xA0,0xA0,0x40,0x40,0x40
"""


def make_consts(step_ticks=2, ball_speed=1, paddle_speed=2, ai_speed=1, ai_rest=4, serve_ticks=45,
                win_score=10):
    return dict(
        LEFT_X=2, RIGHT_X=63,
        STEP_TICKS=step_ticks, BALL_SPEED=ball_speed,
        NEG_BALL_SPEED=(256 - ball_speed) & 0xFF,
        PADDLE_SPEED=paddle_speed, AI_SPEED=ai_speed,
        AI_MASK=max(ai_rest - 1, 0), AI_REST_VAL=0 if ai_rest else 1, SERVE_TICKS=serve_ticks,
        WIN_SCORE=win_score,
    )


def build(**kw):
    """Return (rom_bytes, labels) for the given settings (see make_consts)."""
    return assemble(SRC, make_consts(**kw))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step-ticks", type=int, default=2, help="60 Hz ticks per game step (2 -> 30 steps/s)")
    ap.add_argument("--ball-speed", type=int, default=1, choices=[1, 2, 3], help="ball px per step")
    ap.add_argument("--paddle-speed", type=int, default=2, help="agent paddle px per step")
    ap.add_argument("--ai-speed", type=int, default=1, choices=[1, 2, 3], help="AI paddle px per AI move")
    ap.add_argument("--ai-rest", type=int, default=4, choices=[0, 2, 4, 8, 16, 32],
                    help="AI skips every N-th step (2 -> 0.5 px/step, 4 -> 0.75, 8 -> 0.88, 16 -> 0.94, 32 -> 0.97, 0 -> never rests)")
    ap.add_argument("--win-score", type=int, default=10, help="points needed to win (max 99)")
    ap.add_argument("--serve-ticks", type=int, default=45, help="pause after each point, in 60 Hz ticks")
    default_out = pathlib.Path(__file__).resolve().parent.parent / "roms" / "pong_dql.ch8"
    ap.add_argument("-o", "--out", default=str(default_out))
    a = ap.parse_args()

    rom, _ = build(step_ticks=a.step_ticks, ball_speed=a.ball_speed, paddle_speed=a.paddle_speed,
                   ai_speed=a.ai_speed, ai_rest=a.ai_rest, serve_ticks=a.serve_ticks, win_score=a.win_score)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(rom)
    print(f"wrote {out}  ({len(rom)} bytes, ends at 0x{0x200 + len(rom):03X})  ->  {60 / a.step_ticks:.0f} steps/s")


if __name__ == "__main__":
    main()