"""
Main Game - Virtual Hand Basketball with MediaPipe Hand Control
Integrates all modules: hand detection, finger counting, gesture detection, visual effects
"""

import cv2
import numpy as np
import mediapipe as mp
import random
import math

# Game modules
from column_mapper import ColumnMapper
from renderer import Renderer
from game_timer import GameTimer

# Hand detection modules
from finger_counter import FingerCounter
from gesture_detector import GestureDetector
from visual_effects import VisualEffects


class HandBasketballGame:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Hand detection modules
        self.finger_counter = FingerCounter(smoothing_window=5)
        self.gesture_detector = GestureDetector(stability_frames=8)
        self.visual_effects = VisualEffects()
        
        # Game modules
        self.mapper = ColumnMapper(num_cols=4, smooth=True)
        self.renderer = Renderer(width=960, height=720)
        self.timer = GameTimer(total_time=60)
        
        # Game state
        self.score = 0
        self.lives = 3
        self.balls = []
        self.basket_col = 0
        self.game_over = False
        self.hand_x = 480  # Hand x position
        
        # Ball spawning
        self.last_spawn = 0
        self.base_spawn_interval = 1.5
        self.base_ball_speed = 4
        
        # Effects
        self.bomb_effect_frame = 0
        
        # Current mode values
        self.current_mode_name = "NORMAL"
        self.speed_multiplier = 1.0
        self.score_multiplier = 1.0
        self.finger_count = 0
        
    def spawn_ball(self):
        """Spawn a new ball at random column."""
        col = random.randint(0, 3)
        col_width = 960 / 4
        x = int(col * col_width + col_width / 2)
        
        ball = {
            'x': x,
            'y': 70,
            'col': col,
            'type': 'bomb' if random.random() < 0.15 else 'regular',
            'speed': self.base_ball_speed * self.speed_multiplier
        }
        self.balls.append(ball)
    
    def update_balls(self):
        """Update ball positions and check collisions."""
        to_remove = []
        
        for i, ball in enumerate(self.balls):
            # Update speed based on current multiplier
            ball['speed'] = self.base_ball_speed * self.speed_multiplier
            ball['y'] += ball['speed']
            
            # Check collision at basket level
            if ball['y'] >= 720 - 130:
                if ball['col'] == self.basket_col:
                    if ball['type'] == 'regular':
                        # Apply score multiplier!
                        points = int(10 * self.score_multiplier)
                        self.score += points
                    else:
                        # Bomb caught!
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
    
    def process_hand(self, frame):
        """Process hand detection and return hand position."""
        # Convert to RGB for MediaPipe
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        rgb = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = self.hands.process(rgb)
        
        hand_detected = False
        
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            hand_detected = True
            
            # Draw hand landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Calculate hand center X (using wrist and middle MCP)
            wrist = hand_landmarks.landmark[0]
            middle_mcp = hand_landmarks.landmark[9]
            self.hand_x = int((wrist.x + middle_mcp.x) / 2 * 960)
            
            # Count fingers
            self.finger_count = self.finger_counter.count_fingers(hand_landmarks, handedness)
            
            # Detect game mode
            mode, self.speed_multiplier, self.score_multiplier = self.gesture_detector.detect_mode(self.finger_count)
            mode_info = self.gesture_detector.get_mode_info()
            self.current_mode_name = mode_info['name']
            
            # Draw bounding box with mode color
            self.visual_effects.draw_hand_bounding_box(frame, hand_landmarks, 960, 720, 
                                                        mode_name=self.current_mode_name)
        
        return hand_detected
    
    def run(self):
        print("=" * 60)
        print("🏀 VIRTUAL HAND BASKETBALL GAME 🏀")
        print("=" * 60)
        print("Controls:")
        print("  Move your hand left/right to move the basket")
        print()
        print("Finger Modes:")
        print("  0 fingers (fist) = SLOW-MO (0.5x speed, 0.5x points)")
        print("  1-2 fingers = NORMAL (1x speed, 1x points)")
        print("  3 fingers = FAST (1.5x speed, 1.5x points)")
        print("  4-5 fingers = TURBO (2x speed, 2x points)")
        print()
        print("Press 'q' to quit, 'r' to restart")
        print("=" * 60)
        
        cv2.namedWindow('Hand Basketball', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hand Basketball', 960, 720)
        
        self.timer.start()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (960, 720))
            
            if not self.game_over:
                # Process hand detection
                hand_detected = self.process_hand(frame)
                
                # Map hand position to column
                self.basket_col = self.mapper.map_to_column(self.hand_x, 960)
                
                # Spawn balls (interval affected by speed)
                spawn_interval = self.base_spawn_interval / self.speed_multiplier
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                if current_time - self.last_spawn > spawn_interval:
                    self.spawn_ball()
                    self.last_spawn = current_time
                
                # Update balls
                self.update_balls()
                
                # Check timer
                if self.timer.is_over():
                    self.game_over = True
                
                # Decrease bomb effect
                if self.bomb_effect_frame > 0:
                    self.bomb_effect_frame -= 1
                
                # Apply mode overlay
                frame = self.visual_effects.apply_mode_overlay(frame, self.current_mode_name)
                
                # Render game UI
                self.renderer.render_ui(frame, self.balls, self.basket_col, 
                                       self.score, self.lives, self.timer.time_left(),
                                       self.bomb_effect_frame)
                
                # Draw mode indicator
                self.visual_effects.draw_mode_indicator(frame, self.current_mode_name, 
                                                        self.speed_multiplier, self.score_multiplier,
                                                        position=(750, 30))
                
                # Draw finger count
                self.visual_effects.draw_finger_count(frame, self.finger_count, position=(750, 130))
                
                # Draw speed lines for fast modes
                self.visual_effects.draw_speed_lines(frame, self.current_mode_name)
                
                # Hand position indicator
                cv2.circle(frame, (self.hand_x, 100), 10, (255, 0, 255), -1)
                
                # Detection status
                if not hand_detected:
                    cv2.putText(frame, "No hand detected!", (350, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                self.renderer.draw_game_over(frame, self.score)
            
            cv2.imshow('Hand Basketball', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.restart()
        
        self.cleanup()
    
    def restart(self):
        """Restart the game."""
        self.score = 0
        self.lives = 3
        self.balls = []
        self.basket_col = 0
        self.game_over = False
        self.bomb_effect_frame = 0
        self.timer.reset()
        self.timer.start()
        self.mapper.reset()
        self.finger_counter.reset()
        self.gesture_detector.reset()
        print("Game restarted!")
    
    def cleanup(self):
        """Clean up resources."""
        self.hands.close()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    game = HandBasketballGame()
    game.run()