import cv2
import numpy as np
from column_mapper import ColumnMapper
from renderer import Renderer
from game_timer import GameTimer
import random
import math

class TestTonyWithCamera:
    def __init__(self):
        # Initialize camera with higher resolution
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Tony's modules
        self.mapper = ColumnMapper(num_cols=4, smooth=True)
        self.renderer = Renderer(width=960, height=720)
        self.timer = GameTimer(total_time=60)
        
        # Game state
        self.score = 0
        self.lives = 3
        self.balls = []
        self.basket_col = 0
        self.game_over = False
        
        # Test controls
        self.test_x = 480
        self.auto_mode = True
        
        # Ball spawning
        self.last_spawn = 0
        self.spawn_interval = 1.0
        
        # Bomb effect
        self.bomb_effect_frame = 0
        
    def spawn_ball(self):
        col = random.randint(0, 3)
        col_width = 960 / 4
        x = int(col * col_width + col_width / 2)
        
        ball = {
            'x': x,
            'y': 70,
            'col': col,
            'type': 'bomb' if random.random() < 0.2 else 'regular',
            'speed': 4
        }
        self.balls.append(ball)
    
    def update_balls(self):
        to_remove = []
        
        for i, ball in enumerate(self.balls):
            ball['y'] += ball['speed']
            
            # Check collision at net level
            if ball['y'] >= 720 - 130:
                if ball['col'] == self.basket_col:
                    if ball['type'] == 'regular':
                        self.score += 10
                    else:
                        # Bomb caught! Trigger explosion effect
                        self.bomb_effect_frame = 20
                        self.lives -= 1
                        if self.lives <= 0:
                            self.game_over = True
                else:
                    if ball['type'] == 'regular':
                        self.lives -= 1
                        if self.lives <= 0:
                            self.game_over = True
                
                to_remove.append(i)
            elif ball['y'] > 720:
                to_remove.append(i)
        
        for i in reversed(to_remove):
            del self.balls[i]
    
    def mouse_callback(self, event, x, y, flags, param):
        if not self.auto_mode:
            self.test_x = x
    
    def run(self):
        print("=" * 60)
        print("VIRTUAL HAND BASKETBALL GAME - TEST MODE")
        print("=" * 60)
        print("Controls:")
        print("  A - Toggle AUTO/MOUSE mode")
        print("  R - Restart game")
        print("  Q - Quit")
        print("=" * 60)
        
        cv2.namedWindow('Basketball Game Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Basketball Game Test', 960, 720)
        cv2.setMouseCallback('Basketball Game Test', self.mouse_callback)
        
        self.timer.start()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (960, 720))
            
            if not self.game_over:
                # Auto movement (sinusoidal)
                if self.auto_mode:
                    t = cv2.getTickCount() / cv2.getTickFrequency()
                    self.test_x = int(480 + 380 * math.sin(t * 0.5))
                
                # Map to column
                self.basket_col = self.mapper.map_to_column(self.test_x, 960)
                
                # Spawn balls
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                if current_time - self.last_spawn > self.spawn_interval:
                    self.spawn_ball()
                    self.last_spawn = current_time
                
                self.update_balls()
                
                if self.timer.is_over():
                    self.game_over = True
                
                # Decrease bomb effect frame counter
                if self.bomb_effect_frame > 0:
                    self.bomb_effect_frame -= 1
                
                # Render with bomb effect
                self.renderer.render_ui(frame, self.balls, self.basket_col, 
                                       self.score, self.lives, self.timer.time_left(),
                                       self.bomb_effect_frame)
                
                # Hand position indicator
                cv2.circle(frame, (self.test_x, 120), 12, (255, 0, 255), -1)
                mode_text = "AUTO MODE" if self.auto_mode else "MOUSE MODE"
                cv2.putText(frame, mode_text, (20, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                self.renderer.draw_game_over(frame, self.score)
            
            cv2.imshow('Basketball Game Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.restart()
            elif key == ord('a'):
                self.auto_mode = not self.auto_mode
                print(f"Switched to {'AUTO' if self.auto_mode else 'MOUSE'} mode")
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def restart(self):
        self.score = 0
        self.lives = 3
        self.balls = []
        self.basket_col = 0
        self.game_over = False
        self.bomb_effect_frame = 0
        self.timer.reset()
        self.timer.start()
        self.mapper.reset()
        print("Game restarted!")

if __name__ == '__main__':
    test = TestTonyWithCamera()
    test.run()