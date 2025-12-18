"""
Visual Effects Module - Mode-Based Visual Feedback
Part of: Tony's contribution

Provides visual effects for different game modes including:
- Mode indicator overlay
- Hand outline drawing
- Color-coded feedback
"""

import cv2
import numpy as np


class VisualEffects:
    """
    Applies visual effects to the game frame based on current mode.
    
    Effects per mode:
    - SLOW_MO: Blue tint overlay
    - NORMAL: No effect (clean view)
    - FAST: Yellow/orange glow
    - TURBO: Red glow with intensity
    """
    
    # Mode colors (BGR format for OpenCV)
    MODE_COLORS = {
        'SLOW_MO': (255, 150, 50),    # Blue
        'SLOW-MO': (255, 150, 50),
        'NORMAL': (50, 255, 50),       # Green
        'FAST': (50, 255, 255),        # Yellow
        'TURBO': (50, 50, 255),        # Red
    }
    
    def __init__(self):
        """Initialize visual effects."""
        self.overlay_alpha = 0.15  # Transparency for mode overlay
    
    def draw_mode_indicator(self, frame, mode_name, speed_mult, score_mult, position=(20, 50)):
        """
        Draw mode indicator panel on frame.
        
        Args:
            frame: OpenCV BGR frame
            mode_name: String name of current mode
            speed_mult: Speed multiplier value
            score_mult: Score multiplier value
            position: (x, y) top-left corner of indicator
        """
        x, y = position
        color = self.MODE_COLORS.get(mode_name, (255, 255, 255))
        
        # Draw mode name with glow effect
        cv2.putText(frame, f"Mode: {mode_name}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)  # Shadow
        cv2.putText(frame, f"Mode: {mode_name}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw multipliers
        cv2.putText(frame, f"Speed: {speed_mult}x", (x, y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, f"Speed: {speed_mult}x", (x, y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(frame, f"Points: {score_mult}x", (x, y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, f"Points: {score_mult}x", (x, y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def draw_finger_count(self, frame, finger_count, position=(20, 150)):
        """
        Draw finger count with visual representation.
        
        Args:
            frame: OpenCV BGR frame
            finger_count: Number of fingers (0-5)
            position: (x, y) position to draw
        """
        x, y = position
        
        # Draw number
        cv2.putText(frame, f"Fingers: {finger_count}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, f"Fingers: {finger_count}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw finger icons (circles for extended fingers)
        for i in range(5):
            circle_x = x + 10 + i * 25
            circle_y = y + 25
            if i < finger_count:
                cv2.circle(frame, (circle_x, circle_y), 8, (0, 255, 0), -1)  # Filled
            else:
                cv2.circle(frame, (circle_x, circle_y), 8, (100, 100, 100), 2)  # Outline
        
        return frame
    
    def apply_mode_overlay(self, frame, mode_name):
        """
        Apply color overlay based on game mode.
        
        Args:
            frame: OpenCV BGR frame
            mode_name: Current mode name
            
        Returns:
            Frame with overlay applied
        """
        if mode_name == 'NORMAL':
            return frame  # No overlay for normal mode
        
        color = self.MODE_COLORS.get(mode_name, (255, 255, 255))
        
        # Create colored overlay
        overlay = np.full(frame.shape, color, dtype=np.uint8)
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 1 - self.overlay_alpha, overlay, self.overlay_alpha, 0)
        
        return result
    
    def draw_hand_outline(self, frame, hand_landmarks, mp_drawing, mp_hands, mp_drawing_styles):
        """
        Draw hand landmarks and connections using MediaPipe's drawing utilities.
        
        Args:
            frame: OpenCV BGR frame
            hand_landmarks: MediaPipe hand landmarks
            mp_drawing: MediaPipe drawing solutions
            mp_hands: MediaPipe hands solution
            mp_drawing_styles: MediaPipe drawing styles
        """
        if hand_landmarks is not None:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        return frame
    
    def draw_hand_bounding_box(self, frame, hand_landmarks, width=960, height=720, color=None, mode_name='NORMAL'):
        """
        Draw bounding box around detected hand.
        
        Args:
            frame: OpenCV BGR frame
            hand_landmarks: MediaPipe hand landmarks
            width: Frame width
            height: Frame height
            color: Optional custom color (BGR)
            mode_name: Current mode for color selection
        """
        if hand_landmarks is None:
            return frame
        
        if color is None:
            color = self.MODE_COLORS.get(mode_name, (0, 255, 0))
        
        # Get all landmark positions
        x_coords = [lm.x * width for lm in hand_landmarks.landmark]
        y_coords = [lm.y * height for lm in hand_landmarks.landmark]
        
        # Calculate bounding box with padding
        padding = 20
        x_min = int(max(0, min(x_coords) - padding))
        x_max = int(min(width, max(x_coords) + padding))
        y_min = int(max(0, min(y_coords) - padding))
        y_max = int(min(height, max(y_coords) + padding))
        
        # Draw box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        return frame
    
    def draw_speed_lines(self, frame, mode_name, intensity=0.5):
        """
        Draw speed lines effect for fast modes.
        
        Args:
            frame: OpenCV BGR frame
            mode_name: Current mode
            intensity: Effect intensity (0-1)
        """
        if mode_name not in ['FAST', 'TURBO']:
            return frame
        
        height, width = frame.shape[:2]
        
        # Number of lines based on mode
        num_lines = 5 if mode_name == 'FAST' else 10
        
        color = self.MODE_COLORS.get(mode_name, (255, 255, 255))
        
        # Draw horizontal speed lines on edges
        for i in range(num_lines):
            y = int(height / (num_lines + 1) * (i + 1))
            # Left side
            cv2.line(frame, (0, y), (30, y), color, 2)
            # Right side
            cv2.line(frame, (width - 30, y), (width, y), color, 2)
        
        return frame


def draw_hand_overlay(frame, hand_landmarks, finger_count, mode_name, 
                      mp_drawing=None, mp_hands=None, mp_drawing_styles=None):
    """
    Convenience function to draw all hand-related overlays.
    
    Args:
        frame: OpenCV BGR frame
        hand_landmarks: MediaPipe hand landmarks (or None)
        finger_count: Number of fingers detected
        mode_name: Current game mode name
        mp_drawing, mp_hands, mp_drawing_styles: MediaPipe drawing utilities
    """
    effects = VisualEffects()
    
    # Draw hand landmarks if available and MP utilities provided
    if hand_landmarks is not None and mp_drawing is not None:
        effects.draw_hand_outline(frame, hand_landmarks, mp_drawing, mp_hands, mp_drawing_styles)
        effects.draw_hand_bounding_box(frame, hand_landmarks, mode_name=mode_name)
    
    return frame
