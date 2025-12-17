import cv2
import numpy as np
from column_mapper import ColumnMapper
from renderer import Renderer
from game_timer import GameTimer
from hand_detection import preprocess, segment_hand, get_hand_centroid, draw_debug
import random
import math

class TestTonyWithCamera:
    def __init__(self):
        # Initialize camera with higher resolution
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Tony's modules
        self.mapper = ColumnMapper(num_cols=3, smooth=False)
        self.renderer = Renderer(width=1920, height=1080, num_cols=3)
        self.timer = GameTimer(total_time=60)
        
        # Game state
        self.score = 0
        self.lives = 3
        self.balls = []
        self.basket_col = 0
        self.game_over = False
        
        # Control modes: 'auto', 'mouse', 'hand'
        self.test_x = 960  # Center of 1920
        self.control_mode = 'hand'  # Default to hand tracking
        self.show_debug = False  # Toggle debug visualization
        
        # Smoothing for hand tracking (exponential moving average)
        self.smoothed_x = 960  # Start at center of 1920
        self.smooth_alpha = 0.15  # Lower = smoother but slower response
        
        # Ball spawning
        self.last_spawn = 0
        self.spawn_interval = 3.0
        
        # Bomb effect
        self.bomb_effect_frame = 0
        
    def spawn_ball(self):
        col = random.randint(0, 2)
        col_width = 1920 / 3
        x = int(col * col_width + col_width / 2)
        
        ball = {
            'x': x,
            'y': 70,
            'col': col,
            'type': 'bomb' if random.random() < 0.2 else 'regular',
            'speed': 6
        }
        self.balls.append(ball)
    
    def update_balls(self):
        to_remove = []
        
        for i, ball in enumerate(self.balls):
            ball['y'] += ball['speed']
            
            # Check collision at net level
            if ball['y'] >= 1080 - 130:
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
            elif ball['y'] > 1080:
                to_remove.append(i)
        
        for i in reversed(to_remove):
            del self.balls[i]
    
    def mouse_callback(self, event, x, y, flags, param):
        if self.control_mode == 'mouse':
            self.test_x = x
    
    def run(self):
        print("=" * 60)
        print("VIRTUAL HAND BASKETBALL GAME")
        print("=" * 60)
        print("Controls:")
        print("  H - Hand tracking mode (default)")
        print("  A - Auto movement mode")
        print("  M - Mouse control mode")
        print("  D - Toggle debug visualization")
        print("  R - Restart game")
        print("  Q - Quit")
        print("=" * 60)
        
        cv2.namedWindow('Basketball Game Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Basketball Game Test', 1920, 1080)
        cv2.setMouseCallback('Basketball Game Test', self.mouse_callback)
        
        self.timer.start()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1920, 1080))
            
            if not self.game_over:
                # Get hand position based on control mode
                if self.control_mode == 'hand':
                    # Use hand detection
                    hsv = preprocess(frame)
                    mask = segment_hand(hsv)
                    centroid = get_hand_centroid(mask)
                    
                    if centroid:
                        # Scale x to make edge columns easier to reach
                        # Map from margin to (width-margin) instead of 0 to width  
                        margin = 400  # Large margin - much easier to reach edges
                        raw_x = centroid['x']
                        # Clamp and scale to full width
                        clamped_x = max(margin, min(1920 - margin, raw_x))
                        scaled_x = int((clamped_x - margin) * 1920 / (1920 - 2 * margin))
                        scaled_x = max(0, min(1919, scaled_x))
                        
                        # Apply exponential moving average for smooth tracking
                        self.smoothed_x = self.smooth_alpha * scaled_x + (1 - self.smooth_alpha) * self.smoothed_x
                        self.test_x = int(self.smoothed_x)
                    
                    # Draw debug visualization if enabled
                    if self.show_debug:
                        frame = draw_debug(frame, mask, centroid)
                        
                elif self.control_mode == 'auto':
                    # Auto movement (sinusoidal) - full 1920 width
                    t = cv2.getTickCount() / cv2.getTickFrequency()
                    self.test_x = int(960 + 800 * math.sin(t * 0.5))
                # Mouse mode: test_x is set via mouse_callback
                
                # Map to column
                self.basket_col = self.mapper.map_to_column(self.test_x, 1920)
                
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
                mode_text = f"{self.control_mode.upper()} MODE"
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
            elif key == ord('h'):
                self.control_mode = 'hand'
                print("Switched to HAND tracking mode")
            elif key == ord('a'):
                self.control_mode = 'auto'
                print("Switched to AUTO mode")
            elif key == ord('m'):
                self.control_mode = 'mouse'
                print("Switched to MOUSE mode")
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                print(f"Debug visualization: {'ON' if self.show_debug else 'OFF'}")
        
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