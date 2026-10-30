import random

class Chip8:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)   # seedable -> reproducible DQL runs
        # Bộ nhớ 4KB
        self.memory = [0] * 4096
        
        # 16 thanh ghi 8-bit V0-VF
        self.V = [0] * 16
        
        # Thanh ghi I (16-bit) – trỏ đến địa chỉ bộ nhớ
        self.I = 0
        
        # Program Counter – trỏ đến lệnh hiện tại
        self.pc = 0x200  # ROM luôn bắt đầu tại 0x200
        
        # Stack và Stack Pointer
        self.stack = []
        self.sp = 0
        
        # Timer
        self.delay_timer = 0
        self.sound_timer = 0
        
        # Màn hình 64x32 pixel
        self.display = [[0] * 64 for _ in range(32)]
        
        # Bàn phím 16 phím (0-F)
        self.keypad = [0] * 16
        
        
        # Load font chữ vào bộ nhớ (địa chỉ 0x000 - 0x04F)
        self.load_font()
        # cờ báo frame hoàn chỉnh
        self.draw_flag = True
        # frame_ready: True while the ROM is idle-polling the delay timer (FX07),
        #   i.e. the picture is complete and safe to present. Cleared when a new
        #   step begins (FX15). Presenting only then removes paddle flicker.
        self.frame_ready = False
        self.frame_ended = False
        self.halted = False   # True once the ROM jumps to itself (game over)
    
    def load_font(self):
        """Load bộ font chữ 4x5 pixel cho các ký tự 0-F"""
        fontset = [
            0xF0, 0x90, 0x90, 0x90, 0xF0,  # 0
            0x20, 0x60, 0x20, 0x20, 0x70,  # 1
            0xF0, 0x10, 0xF0, 0x80, 0xF0,  # 2
            0xF0, 0x10, 0xF0, 0x10, 0xF0,  # 3
            0x90, 0x90, 0xF0, 0x10, 0x10,  # 4
            0xF0, 0x80, 0xF0, 0x10, 0xF0,  # 5
            0xF0, 0x80, 0xF0, 0x90, 0xF0,  # 6
            0xF0, 0x10, 0x20, 0x40, 0x40,  # 7
            0xF0, 0x90, 0xF0, 0x90, 0xF0,  # 8
            0xF0, 0x90, 0xF0, 0x10, 0xF0,  # 9
            0xF0, 0x90, 0xF0, 0x90, 0x90,  # A
            0xE0, 0x90, 0xE0, 0x90, 0xE0,  # B
            0xF0, 0x80, 0x80, 0x80, 0xF0,  # C
            0xE0, 0x90, 0x90, 0x90, 0xE0,  # D
            0xF0, 0x80, 0xF0, 0x80, 0xF0,  # E
            0xF0, 0x80, 0xF0, 0x80, 0x80,  # F
        ]
        for i, byte in enumerate(fontset):
            self.memory[i] = byte
    
    def load_rom(self, path):
        """Đọc file ROM và nạp vào bộ nhớ tại 0x200"""
        with open(path, "rb") as f:
            rom = f.read()
            for i, byte in enumerate(rom):
                self.memory[0x200 + i] = byte
    
    def cycle(self):
        """Thực thi một chu kỳ: fetch → decode → execute"""
        # FETCH: đọc 2 byte từ bộ nhớ tại PC
        opcode = (self.memory[self.pc] << 8) | self.memory[self.pc + 1]
        self.pc += 2

        # DECODE: tách opcode
        x = (opcode & 0x0F00) >> 8
        y = (opcode & 0x00F0) >> 4
        n = opcode & 0x000F
        nn = opcode & 0x00FF
        nnn = opcode & 0x0FFF
        
        # EXECUTE
        first = (opcode & 0xF000) >> 12
        
        if opcode == 0x00E0:  # CLS – xóa màn hình
            self.display = [[0] * 64 for _ in range(32)]
            self.draw_flag = True
            self.frame_ready = False   # CLS starts a new picture
        elif opcode == 0x00EE:  # RET – trở về từ subroutine
            self.pc = self.stack.pop()
        elif first == 0x1:  # JP addr
            if nnn == self.pc - 2:   # jump-to-self = ROM finished
                self.halted = True
            self.pc = nnn
        elif first == 0x2:  # CALL addr
            self.stack.append(self.pc)
            self.pc = nnn
        elif first == 0x3:  # SE Vx, byte
            if self.V[x] == nn:
                self.pc += 2
        elif first == 0x4:  # SNE Vx, byte
            if self.V[x] != nn:
                self.pc += 2
        elif first == 0x5:  # SE Vx, Vy
            if self.V[x] == self.V[y]:
                self.pc += 2
        elif first == 0x6:  # LD Vx, byte
            self.V[x] = nn
        elif first == 0x7:  # ADD Vx, byte
            self.V[x] = (self.V[x] + nn) & 0xFF
        elif first == 0x8:  # ALU operations
            if n == 0x0:    # LD Vx, Vy
                self.V[x] = self.V[y]
            elif n == 0x1:  # OR
                self.V[x] |= self.V[y]
            elif n == 0x2:  # AND
                self.V[x] &= self.V[y]
            elif n == 0x3:  # XOR
                self.V[x] ^= self.V[y]
            elif n == 0x4:  # ADD (VF = carry, written last)
                total = self.V[x] + self.V[y]
                self.V[x] = total & 0xFF
                self.V[0xF] = 1 if total > 0xFF else 0
            elif n == 0x5:  # SUB (VF = NOT borrow, written last)
                flag = 1 if self.V[x] >= self.V[y] else 0
                self.V[x] = (self.V[x] - self.V[y]) & 0xFF
                self.V[0xF] = flag
            elif n == 0x6:  # SHR
                flag = self.V[x] & 0x1
                self.V[x] >>= 1
                self.V[0xF] = flag
            elif n == 0x7:  # SUBN
                flag = 1 if self.V[y] >= self.V[x] else 0
                self.V[x] = (self.V[y] - self.V[x]) & 0xFF
                self.V[0xF] = flag
            elif n == 0xE:  # SHL
                flag = (self.V[x] & 0x80) >> 7
                self.V[x] = (self.V[x] << 1) & 0xFF
                self.V[0xF] = flag
        elif first == 0x9:  # SNE Vx, Vy
            if self.V[x] != self.V[y]:
                self.pc += 2
        elif first == 0xA:  # LD I, addr
            self.I = nnn
        elif first == 0xB:  # JP V0, addr
            self.pc = nnn + self.V[0]
        elif first == 0xC:  # RND Vx, byte
            self.V[x] = self.rng.randint(0, 255) & nn
        elif first == 0xD:  # DRW Vx, Vy, nibble – VẼ SPRITE
            self.V[0xF] = 0
            for row in range(n):
                sprite = self.memory[self.I + row]
                for col in range(8):
                    if sprite & (0x80 >> col):
                        px = (self.V[x] + col) % 64
                        py = (self.V[y] + row) % 32
                        if self.display[py][px] == 1:
                            self.V[0xF] = 1
                        self.display[py][px] ^= 1
            self.draw_flag = True
        elif first == 0xE:
            if nn == 0x9E:  # SKP Vx – nếu phím được nhấn
                if self.keypad[self.V[x]]:
                    self.pc += 2
            elif nn == 0xA1:  # SKNP Vx – nếu phím không được nhấn
                if not self.keypad[self.V[x]]:
                    self.pc += 2
        elif first == 0xF:
            if nn == 0x07:  # LD Vx, DT  (ROM polling the timer == frame finished)
                self.V[x] = self.delay_timer
                self.frame_ready = True
            elif nn == 0x0A:  # LD Vx, K – chờ nhấn phím
                pressed = False
                for i in range(16):
                    if self.keypad[i]:
                        self.V[x] = i
                        pressed = True
                        break
                if not pressed:
                    self.pc -= 2  # Chờ tiếp
            elif nn == 0x15:  # LD DT, Vx
                self.delay_timer = self.V[x]
                self.frame_ready = False   # new step started, picture is being rebuilt
                self.frame_ended = True
            elif nn == 0x18:  # LD ST, Vx
                self.sound_timer = self.V[x]
            elif nn == 0x1E:  # ADD I, Vx
                self.I += self.V[x]
            elif nn == 0x29:  # LD F, Vx – font chữ
                self.I = self.V[x] * 5
            elif nn == 0x33:  # BCD – binary coded decimal
                value = self.V[x]
                self.memory[self.I] = value // 100
                self.memory[self.I + 1] = (value // 10) % 10
                self.memory[self.I + 2] = value % 10
            elif nn == 0x55:  # LD [I], Vx – lưu thanh ghi vào bộ nhớ
                for i in range(x + 1):
                    self.memory[self.I + i] = self.V[i]
            elif nn == 0x65:  # LD Vx, [I] – đọc từ bộ nhớ vào thanh ghi
                for i in range(x + 1):
                    self.V[i] = self.memory[self.I + i]
    
    def tick_timers(self):
        """Giảm timer ở tần số 60Hz"""
        if self.delay_timer > 0:
            self.delay_timer -= 1
        if self.sound_timer > 0:
            self.sound_timer -= 1
