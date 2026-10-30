import pygame


class Display:
    SCALE = 10
    
    def __init__(self):
        pygame.init()
        self.width = 64 * self.SCALE
        self.height = 32 * self.SCALE
        self.screen = pygame.display.set_mode((self.width, self.height),pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.display.set_caption("CHIP-8 Emulator")
        self.clock = pygame.time.Clock()   # ← Đảm bảo có dòng này
    
    def render(self, chip8):
        self.screen.fill((0, 0, 0))
        for y in range(32):
            for x in range(64):
                if chip8.display[y][x]:
                    pygame.draw.rect(
                        self.screen,
                        (255, 255, 255),
                        (x * self.SCALE, y * self.SCALE,
                         self.SCALE, self.SCALE)
                    )
        pygame.display.flip()
    
    def tick(self, fps=60):
        """Giới hạn FPS của game."""
        self.clock.tick(fps)
    def close(self):
        """Đóng cửa sổ pygame và giải phóng tài nguyên."""
        pygame.quit()