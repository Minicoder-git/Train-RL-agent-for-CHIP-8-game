import pygame


class Keyboard:
    """
    Quản lý bàn phím CHIP-8 (16 phím hex 0-F).
    
    Ánh xạ bàn phím vật lý → bàn phím CHIP-8:
    
        CHIP-8 Keypad:          Bàn phím PC:
        1 2 3 C                 1 2 3 4
        4 5 6 D                 Q W E R
        7 8 9 E                 A S D F
        A 0 B F                 Z X C V
    
    Cách nhớ: 4 hàng của CHIP-8 tương ứng 4 hàng phím trên PC.
    """
    
    # Ánh xạ phím vật lý → mã phím CHIP-8
    KEYMAP = {
        # Hàng 1: 1 2 3 C
        pygame.K_1: 0x1,
        pygame.K_2: 0x2,
        pygame.K_3: 0x3,
        pygame.K_4: 0xC,
        
        # Hàng 2: 4 5 6 D
        pygame.K_q: 0x4,
        pygame.K_w: 0x5,
        pygame.K_e: 0x6,
        pygame.K_r: 0xD,
        
        # Hàng 3: 7 8 9 E
        pygame.K_a: 0x7,
        pygame.K_s: 0x8,
        pygame.K_d: 0x9,
        pygame.K_f: 0xE,
        
        # Hàng 4: A 0 B F
        pygame.K_z: 0xA,
        pygame.K_x: 0x0,
        pygame.K_c: 0xB,
        pygame.K_v: 0xF,
    }
    
    # Bảng tra ngược: mã CHIP-8 → tên phím hiển thị (để debug)
    KEY_NAMES = {
        0x1: "1", 0x2: "2", 0x3: "3", 0xC: "4",
        0x4: "Q", 0x5: "W", 0x6: "E", 0xD: "R",
        0x7: "A", 0x8: "S", 0x9: "D", 0xE: "F",
        0xA: "Z", 0x0: "X", 0xB: "C", 0xF: "V",
    }
    
    def __init__(self, chip8):
        """
        Khởi tạo bàn phím.
        
        Args:
            chip8: Instance của Chip8 (để ghi trực tiếp vào chip8.keypad)
        """
        self.chip8 = chip8
        self.running = True
        # Theo dõi phím vừa được nhấn (cho các game cần edge detection)
        self.last_key_pressed = None
    
    def handle_events(self):
        """
        Xử lý tất cả sự kiện pygame liên quan đến bàn phím và cửa sổ.
        
        Returns:
            bool: False nếu người dùng đóng cửa sổ, True nếu tiếp tục chạy.
        """
        for event in pygame.event.get():
            # Người dùng đóng cửa sổ
            if event.type == pygame.QUIT:
                self.running = False
                return False
            
            # Phím được nhấn xuống
            elif event.type == pygame.KEYDOWN:
                self._on_key_down(event.key)
            
            # Phím được nhả ra
            elif event.type == pygame.KEYUP:
                self._on_key_up(event.key)
        
        return True
    
    def _on_key_down(self, pygame_key):
        """Xử lý khi một phím được nhấn xuống."""
        if pygame_key in self.KEYMAP:
            chip8_key = self.KEYMAP[pygame_key]
            self.chip8.keypad[chip8_key] = 1
            self.last_key_pressed = chip8_key
            

        # Phím ESC để thoát
        elif pygame_key == pygame.K_ESCAPE:
            self.running = False
    
    def _on_key_up(self, pygame_key):
        """Xử lý khi một phím được nhả ra."""
        if pygame_key in self.KEYMAP:
            chip8_key = self.KEYMAP[pygame_key]
            self.chip8.keypad[chip8_key] = 0
    
    def get_pressed_key(self):
        """
        Trả về phím CHIP-8 đang được nhấn (nếu có), ngược lại trả về None.
        Hữu ích cho opcode 0xFX0A (chờ nhấn phím).
        """
        for i in range(16):
            if self.chip8.keypad[i]:
                return i
        return None
    
    def clear(self):
        """Reset toàn bộ trạng thái bàn phím."""
        for i in range(16):
            self.chip8.keypad[i] = 0
        self.last_key_pressed = None
    
    def is_running(self):
        """Kiểm tra xem game còn chạy không."""
        return self.running