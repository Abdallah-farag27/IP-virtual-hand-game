import cv2
import time
import random
import math
from hand_detector import HandDetector
from column_mapper import ColumnMapper
from renderer import Renderer
from game_timer import GameTimer
import config

class VirtualHandGame:
    """
    Virtual Hand Basketball Game with improved hand detection.
    
    Uses classical image processing techniques for robust hand tracking:
    - ROI restriction
    - Contour area filtering
    - Temporal smoothing
    - Distance transform palm detection
    - 3-column layout
    """
    
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        
        # Initialize hand detector with config settings
        self.detector = HandDetector(
            roi_height_ratio=config.ROI_HEIGHT_RATIO,
            min_area=config.MIN_HAND_AREA,
            max_area=config.MAX_HAND_AREA
        )
        self.detector.smoothing_alpha = config.SMOOTHING_ALPHA
        
        # Initialize column mapper
        self.mapper = ColumnMapper(
            num_cols=config.NUM_COLUMNS,
            smooth=config.ENABLE_COLUMN_SMOOTHING,
            max_jump=config.MAX_COLUMN_JUMP
        )
        
        # Initialize renderer
        self.renderer = Renderer(
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            num_cols=config.NUM_COLUMNS
        )
        
        # Initialize timer
        self.timer = GameTimer(total_time=config.GAME_DURATION)
        
        # Game state
        self.score = 0
        self.lives = config.STARTING_LIVES
        self.balls = []
        self.basket_col = config.NUM_COLUMNS // 2  # Start at center
        self.game_over = False
        
        # Ball spawning
        self.last_spawn = 0
        self.spawn_interval = config.BALL_SPAWN_INTERVAL
        
        # Bomb effect
        self.bomb_effect_frame = 0
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = cv2.getTickCount()
    
    def calculate_fps(self):
        """Calculate and update FPS."""
        self.frame_count += 1
        if self.frame_count >= 10:
            current_time = cv2.getTickCount()
            elapsed = (current_time - self.fps_start_time) / cv2.getTickFrequency()
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def spawn_ball(self):
        """Spawn a new ball or bomb."""
        col = random.randint(0, config.NUM_COLUMNS - 1)
        col_width = config.CAMERA_WIDTH / config.NUM_COLUMNS
        x = int(col * col_width + col_width / 2)
        
        ball = {
            'x': x,
            'y': 70,
            'col': col,
            'type': 'bomb' if random.random() < config.BOMB_PROBABILITY else 'regular',
            'speed': config.BALL_SPEED
        }
        self.balls.append(ball)
    
    def update_balls(self):
        """Update ball positions and check collisions."""
        to_remove = []
        
        for i, ball in enumerate(self.balls):
            ball['y'] += ball['speed']
            
            # Check collision at net level
            if ball['y'] >= config.CAMERA_HEIGHT - 130:
                if ball['col'] == self.basket_col:
                    # Ball caught!
                    if ball['type'] == 'regular':
                        self.score += config.POINTS_PER_BALL
                    else:
                        # Bomb caught! Trigger explosion
                        self.bomb_effect_frame = config.BOMB_EFFECT_FRAMES
                        self.lives -= config.LIVES_LOST_PER_BOMB
                        if self.lives <= 0:
                            self.game_over = True
                else:
                    # Ball missed!
                    if ball['type'] == 'regular':
                        self.lives -= config.LIVES_LOST_PER_MISS
                        if self.lives <= 0:
                            self.game_over = True
                
                to_remove.append(i)
            elif ball['y'] > config.CAMERA_HEIGHT:
                to_remove.append(i)
        
        # Remove collected/missed balls
        for i in reversed(to_remove):
            del self.balls[i]
    
    def draw_debug_info(self, frame, hand_info):
        """Draw debug information on frame."""
        if not config.SHOW_DEBUG_INFO:
            return
        
        y_pos = 80
        line_height = 25
        
        # ROI boundary
        roi_y = hand_info['roi_start_y']
        cv2.line(frame, (0, roi_y), (config.CAMERA_WIDTH, roi_y), config.ROI_COLOR, 2)
        cv2.putText(frame, "ROI", (10, roi_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.ROI_COLOR, 1)
        
        # Hand position
        if hand_info['detected']:
            cx, cy = hand_info['x'], hand_info['y']
            
            # Draw hand contour
            if hand_info['contour'] is not None:
                contour_offset = hand_info['contour'].copy()
                contour_offset[:, :, 1] += roi_y
                cv2.drawContours(frame, [contour_offset], -1, config.HAND_COLOR, 2)
            
            # Draw hand center
            cv2.circle(frame, (cx, cy), 10, config.HAND_COLOR, -1)
            cv2.circle(frame, (cx, cy), 15, (255, 255, 255), 2)
            
            # Show palm radius if using palm detection
            if config.USE_PALM_DETECTION and 'palm_radius' in hand_info:
                palm_radius = int(hand_info['palm_radius'])
                cv2.circle(frame, (cx, cy), palm_radius, (255, 0, 255), 1)
            
            # Position text
            cv2.putText(frame, f"Hand: ({cx}, {cy})", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.TEXT_COLOR, 1)
            y_pos += line_height
        
        # Basket column
        cv2.putText(frame, f"Basket: Col {self.basket_col}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.TEXT_COLOR, 1)
    
    def game_loop(self):
        """Main game loop."""
        print("=" * 80)
        print("VIRTUAL HAND BASKETBALL GAME")
        print("=" * 80)
        print(f"Settings:")
        print(f"  - Columns: {config.NUM_COLUMNS}")
        print(f"  - Duration: {config.GAME_DURATION}s")
        print(f"  - Detection: {'Palm (Distance Transform)' if config.USE_PALM_DETECTION else 'Centroid'}")
        print(f"  - Smoothing: α={config.SMOOTHING_ALPHA}")
        print(f"  - YCrCb: {'Enabled' if config.USE_YCRCB else 'Disabled (HSV only)'}")
        print()
        print("Controls:")
        print("  Q - Quit")
        print("  R - Restart")
        print("  D - Toggle debug info")
        print()
        print("Move your hand in the LOWER half of the camera to control the basket!")
        print("Catch basketballs (+10 points), avoid bombs (-1 life)!")
        print("=" * 80)
        
        self.timer.start()
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT))
            
            if not self.game_over:
                # Detect hand
                if config.USE_PALM_DETECTION:
                    hand_info = self.detector.detect_hand_with_palm_center(
                        frame, use_smoothing=True
                    )
                else:
                    hand_info = self.detector.detect_hand(
                        frame, use_smoothing=True
                    )
                
                # Map to column
                if hand_info['detected'] and hand_info['x'] is not None:
                    self.basket_col = self.mapper.map_to_column(hand_info['x'], config.CAMERA_WIDTH)
                
                # Spawn balls
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                if current_time - self.last_spawn > self.spawn_interval:
                    self.spawn_ball()
                    self.last_spawn = current_time
                
                # Update balls
                self.update_balls()
                
                # Decrease bomb effect frame counter
                if self.bomb_effect_frame > 0:
                    self.bomb_effect_frame -= 1
                
                # Check game over
                if self.timer.is_over():
                    self.game_over = True
                
                # Render game
                self.renderer.render_ui(
                    frame, self.balls, self.basket_col,
                    self.score, self.lives, self.timer.time_left(),
                    self.bomb_effect_frame
                )
                
                # Draw debug info
                self.draw_debug_info(frame, hand_info)
                
                # Show FPS
                if config.SHOW_FPS:
                    self.calculate_fps()
                    cv2.putText(frame, f"FPS: {self.fps:.1f}", (config.CAMERA_WIDTH - 120, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.TEXT_COLOR, 2)
            else:
                # Game over screen
                self.renderer.draw_game_over(frame, self.score)
            
            # Show frame
            cv2.imshow(config.WINDOW_NAME, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and self.game_over:
                self.restart()
            elif key == ord('d'):
                config.SHOW_DEBUG_INFO = not config.SHOW_DEBUG_INFO
                print(f"Debug info: {'ON' if config.SHOW_DEBUG_INFO else 'OFF'}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 80)
        print("GAME OVER!")
        print(f"Final Score: {self.score}")
        print("=" * 80)
    
    def restart(self):
        """Restart the game."""
        self.score = 0
        self.lives = config.STARTING_LIVES
        self.balls = []
        self.basket_col = config.NUM_COLUMNS // 2
        self.game_over = False
        self.bomb_effect_frame = 0
        self.timer.reset()
        self.timer.start()
        self.mapper.reset()
        self.detector.reset()
        print("\nGame restarted!")

if __name__ == '__main__':
    game = VirtualHandGame()
    game.game_loop()