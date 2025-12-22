import cv2
import numpy as np
import math

class Renderer:
    def __init__(self, width=960, height=720, num_cols=3):
        self.width = width
        self.height = height
        self.num_cols = num_cols
        self.col_width = width / num_cols
        
        self.load_images()
        
        self.TEXT_COLOR = (255, 255, 255)
        self.DIVIDER_COLOR = (150, 150, 150)
        
    def load_images(self):
        self.basketball = np.zeros((60, 60, 3), dtype=np.uint8)
        cv2.circle(self.basketball, (30, 30), 28, (0, 100, 255), -1)
        cv2.ellipse(self.basketball, (30, 30), (28, 28), 0, -45, 45, (0, 0, 0), 2)
        cv2.ellipse(self.basketball, (30, 30), (28, 28), 90, -45, 45, (0, 0, 0), 2)
        cv2.circle(self.basketball, (30, 30), 28, (0, 0, 0), 2)
        
        self.bomb = np.zeros((60, 60, 3), dtype=np.uint8)
        cv2.circle(self.bomb, (30, 35), 22, (30, 30, 30), -1)
        cv2.line(self.bomb, (30, 13), (30, 3), (150, 150, 150), 3)
        cv2.circle(self.bomb, (30, 3), 4, (0, 0, 255), -1)
        cv2.circle(self.bomb, (30, 35), 22, (0, 0, 0), 2)
        
        self.net_width = 140
        self.net_height = 80
        self.net = np.zeros((self.net_height, self.net_width, 3), dtype=np.uint8)
        
        rim_y = 15
        cv2.ellipse(self.net, (self.net_width//2, rim_y), (60, 8), 0, 0, 180, (0, 140, 255), -1)
        cv2.ellipse(self.net, (self.net_width//2, rim_y), (60, 8), 0, 0, 180, (0, 80, 200), 3)
        
        net_spacing = 12
        for i in range(10, self.net_width-10, net_spacing):
            cv2.line(self.net, (i, rim_y), (self.net_width//2, self.net_height-5), (255, 255, 255), 1)
        
        for y in range(rim_y+10, self.net_height-10, 15):
            cv2.ellipse(self.net, (self.net_width//2, y), (int(55*(1-(y-rim_y)/60)), 3), 0, 0, 180, (255, 255, 255), 1)
    
    def draw_column_dividers(self, frame):
        for i in range(1, self.num_cols):
            x = int(i * self.col_width)
            cv2.line(frame, (x, 0), (x, self.height), self.DIVIDER_COLOR, 2)
    
    def draw_net(self, frame, basket_col):
        x_center = int(basket_col * self.col_width + self.col_width / 2)
        y_bottom = self.height - 50
        
        x1 = x_center - self.net_width // 2
        y1 = y_bottom - self.net_height
        
        # Ensure net stays in bounds
        x1 = max(0, min(x1, self.width - self.net_width))
        
        roi = frame[y1:y1+self.net_height, x1:x1+self.net_width]
        mask = cv2.cvtColor(self.net, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        
        net_overlay = cv2.bitwise_and(self.net, self.net, mask=mask)
        background = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
        roi[:] = cv2.add(background, net_overlay)
    
    def draw_balls(self, frame, balls):
        for ball in balls:
            x = int(ball['x'])
            y = int(ball['y'])
            ball_type = ball.get('type', 'regular')
            
            # Choose ball image
            ball_img = self.bomb if ball_type == 'bomb' else self.basketball
            ball_size = ball_img.shape[0]
            half_size = ball_size // 2
            
            # Ensure ball is within frame bounds
            if (x - half_size >= 0 and x + half_size <= self.width and
                y - half_size >= 0 and y + half_size <= self.height):
                
                y1, y2 = y - half_size, y + half_size
                x1, x2 = x - half_size, x + half_size
                
                # Overlay ball with transparency
                roi = frame[y1:y2, x1:x2]
                mask = cv2.cvtColor(ball_img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                
                ball_overlay = cv2.bitwise_and(ball_img, ball_img, mask=mask)
                background = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
                roi[:] = cv2.add(background, ball_overlay)
    
    def draw_ui_info(self, frame, score, lives, time_left, finger_count=None, mode_name="NORMAL", speed_multiplier=None):
        # Dark background panel
        cv2.rectangle(frame, (0, 0), (self.width, 60), (40, 40, 40), -1)
        
        # Score
        cv2.putText(frame, f"Score: {score}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.TEXT_COLOR, 2)
        
        # Lives (with heart symbols)
        lives_text = f"Lives: {lives}"
        cv2.putText(frame, lives_text, (self.width//2 - 80, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.TEXT_COLOR, 2)
                    
        # Mode & Fingers
        mode_color = (0, 255, 0)
        if mode_name == "SLOW-MO": mode_color = (255, 255, 0)
        elif mode_name == "FAST": mode_color = (0, 165, 255)
        elif mode_name == "TURBO": mode_color = (0, 0, 255)
        
        if finger_count is not None:
             cv2.putText(frame, f"Fingers: {finger_count}", (self.width//2 + 100, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
             # Display mode below fingers
             cv2.putText(frame, f"MODE: {mode_name}", (self.width//2 + 100, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
             
             # Display speed below mode
             if speed_multiplier is not None:
                 cv2.putText(frame, f"Speed: {speed_multiplier:.1f}x", (self.width//2 + 100, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Timer
        time_str = f"Time: {int(time_left)}s"
        time_size = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.putText(frame, time_str, (self.width - time_size[0] - 20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.TEXT_COLOR, 2)
    
    def draw_bomb_effect(self, frame, basket_col, effect_frame):
        """Draw explosion effect when bomb is caught"""
        if effect_frame > 0:
            # Calculate explosion center
            x_center = int(basket_col * self.col_width + self.col_width / 2)
            y_center = self.height - 90
            
            # Explosion radius grows then shrinks
            max_radius = 80
            if effect_frame > 10:
                radius = int((20 - effect_frame) * max_radius / 10)
            else:
                radius = int(effect_frame * max_radius / 10)
            
            # Red flash overlay
            intensity = int(255 * (effect_frame / 20))
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, intensity), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Draw explosion circles
            cv2.circle(frame, (x_center, y_center), radius, (0, 0, 255), 3)
            cv2.circle(frame, (x_center, y_center), int(radius * 0.7), (0, 100, 255), 2)
            cv2.circle(frame, (x_center, y_center), int(radius * 0.4), (0, 165, 255), 2)
            
            # Explosion particles (lines radiating out)
            
            for angle in range(0, 360, 30):
                rad = math.radians(angle)
                x1 = int(x_center + radius * 0.5 * math.cos(rad))
                y1 = int(y_center + radius * 0.5 * math.sin(rad))
                x2 = int(x_center + radius * math.cos(rad))
                y2 = int(y_center + radius * math.sin(rad))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            
            # "BOOM!" text
            if effect_frame > 5:
                text = "BOOM!"
                font = cv2.FONT_HERSHEY_SIMPLEX 
                text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
                text_x = x_center - text_size[0] // 2
                text_y = y_center - radius - 20
                cv2.putText(frame, text, (text_x, text_y), font, 1.5, (0, 0, 255), 4)
    
    def render_ui(self, frame, balls, basket_col, score, lives, time_left, bomb_effect_frame=0, finger_count=None, mode_name="NORMAL", speed_multiplier=None):
        # Draw columns
        self.draw_column_dividers(frame)
        
        # Draw net
        self.draw_net(frame, basket_col)
        
        # Draw balls
        self.draw_balls(frame, balls)
        
        # Draw bomb effect
        self.draw_bomb_effect(frame, basket_col, bomb_effect_frame)
        
        # Draw top info bar
        self.draw_ui_info(frame, score, lives, time_left, finger_count, mode_name, speed_multiplier)
            
    def draw_hand(self, frame, hand_info):
        if not hand_info['detected']:
            return
            
        roi_y = hand_info['roi_start_y']
        
        # 1. Draw Contour
        if hand_info['contour'] is not None:
            contour_offset = hand_info['contour'].copy()
            contour_offset[:, :, 1] += roi_y
            # Green contour
            cv2.drawContours(frame, [contour_offset], -1, (0, 255, 0), 2)
            
        # 2. Draw Center Point
        cx, cy = hand_info['x'], hand_info['y']
        # Green center dot
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
        # White ring outline
        cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
        
        # 3. Draw Palm Radius (optional visual aid)
        if 'palm_radius' in hand_info:
            radius = int(hand_info['palm_radius'])
            cv2.circle(frame, (cx, cy), radius, (255, 0, 255), 1)
            
        # 4. MediaPipe Landmarks
        if hand_info.get('type') == 'mediapipe' and hand_info.get('landmarks'):
             import mediapipe as mp
             mp_draw = mp.solutions.drawing_utils
             mp_hands = mp.solutions.hands
             mp_draw.draw_landmarks(frame, hand_info['landmarks'], mp_hands.HAND_CONNECTIONS)

    
    def draw_game_over(self, frame, final_score, is_win=False):
        # Semi-transparent dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Text based on win/loss
        if is_win:
            text = "YOU WON!"
            color = (0, 255, 0) # Green
        else:
            text = "GAME OVER"
            color = (0, 0, 255) # Red

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 2.5, 4)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = self.height // 2 - 80
        cv2.putText(frame, text, (text_x, text_y), font, 2.5, color, 5)
        
        # Final score
        score_text = f"Final Score: {final_score}"
        score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        score_x = (self.width - score_size[0]) // 2
        score_y = text_y + 100
        cv2.putText(frame, score_text, (score_x, score_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.TEXT_COLOR, 3)
        
        # Instructions
        inst_text = "Press 'R' for Menu or 'Q' to Quit"
        inst_size = cv2.getTextSize(inst_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        inst_x = (self.width - inst_size[0]) // 2
        inst_y = score_y + 70
        cv2.putText(frame, inst_text, (inst_x, inst_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.TEXT_COLOR, 2)
    def draw_start_screen(self, frame):
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Title
        title = "SELECT GAME MODE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_size = cv2.getTextSize(title, font, 2, 4)[0]
        cv2.putText(frame, title, ((self.width - title_size[0])//2, 200), font, 2, (255, 255, 255), 4)
        
        # Options
        opt1 = "1. Classic (Image Processing)"
        opt2 = "2. AI (MediaPipe)"
        
        cv2.putText(frame, opt1, (200, 350), font, 1.2, (0, 255, 255), 2)
        cv2.putText(frame, opt2, (200, 450), font, 1.2, (0, 255, 0), 2)
        
        # Instructions
        inst = "Press 1 or 2 to start"
        inst_size = cv2.getTextSize(inst, font, 1, 2)[0]
        cv2.putText(frame, inst, ((self.width - inst_size[0])//2, 600), font, 1, (200, 200, 200), 2)
