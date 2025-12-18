"""
Gesture Detector Module - Maps Finger Count to Game Modes
Part of: Fararg & Tony's contribution

Detects game mode based on finger count and provides speed/score multipliers.
Implements gesture stability to prevent flickering between modes.
"""

from enum import Enum
from collections import deque


class GameMode(Enum):
    """Game modes based on finger count."""
    SLOW_MO = "slow_mo"     # 0 fingers (fist) - easier but fewer points
    NORMAL = "normal"        # 1-2 fingers - standard gameplay
    FAST = "fast"            # 3 fingers - faster, more points
    TURBO = "turbo"          # 4-5 fingers - fastest, most points


class GestureDetector:
    """
    Detects game mode from finger count with stability.
    
    Game Modes:
    - SLOW_MO (0 fingers): 0.5x speed, 0.5x points - for beginners/practice
    - NORMAL (1-2 fingers): 1.0x speed, 1.0x points - standard play
    - FAST (3 fingers): 1.5x speed, 1.5x points - challenge mode
    - TURBO (4-5 fingers): 2.0x speed, 2.0x points - expert mode
    
    Includes stability mechanism to require consistent gesture before switching modes.
    """
    
    # Mode configuration: (speed_multiplier, score_multiplier, display_name, color_rgb)
    MODE_CONFIG = {
        GameMode.SLOW_MO: (0.5, 0.5, "SLOW-MO", (255, 150, 50)),    # Blue-ish
        GameMode.NORMAL:  (1.0, 1.0, "NORMAL", (50, 255, 50)),       # Green
        GameMode.FAST:    (1.5, 1.5, "FAST", (50, 255, 255)),        # Yellow
        GameMode.TURBO:   (2.0, 2.0, "TURBO", (50, 50, 255)),        # Red
    }
    
    def __init__(self, stability_frames=10):
        """
        Initialize gesture detector with stability requirement.
        
        Args:
            stability_frames: Number of consistent frames required before mode switch
        """
        self.stability_frames = stability_frames
        self.current_mode = GameMode.NORMAL
        self.pending_mode = GameMode.NORMAL
        self.pending_count = 0
        self.mode_history = deque(maxlen=30)  # Track mode history for analytics
    
    def detect_mode(self, finger_count):
        """
        Detect game mode from finger count with stability.
        
        Args:
            finger_count: Number of extended fingers (0-5)
            
        Returns:
            tuple: (GameMode, speed_multiplier, score_multiplier)
        """
        # Map finger count to mode
        new_mode = self._finger_count_to_mode(finger_count)
        
        # Stability check - require consistent gesture before switching
        if new_mode == self.pending_mode:
            self.pending_count += 1
        else:
            self.pending_mode = new_mode
            self.pending_count = 1
        
        # Switch mode if stable for enough frames
        if self.pending_count >= self.stability_frames:
            if self.current_mode != new_mode:
                self.current_mode = new_mode
                self.mode_history.append(new_mode)
        
        # Get multipliers for current mode
        speed_mult, score_mult, _, _ = self.MODE_CONFIG[self.current_mode]
        
        return self.current_mode, speed_mult, score_mult
    
    def _finger_count_to_mode(self, finger_count):
        """
        Map finger count to game mode.
        
        Args:
            finger_count: Number of fingers (0-5)
            
        Returns:
            GameMode: Corresponding game mode
        """
        if finger_count == 0:
            return GameMode.SLOW_MO
        elif finger_count <= 2:
            return GameMode.NORMAL
        elif finger_count == 3:
            return GameMode.FAST
        else:  # 4 or 5 fingers
            return GameMode.TURBO
    
    def get_mode_info(self):
        """
        Get complete information about current mode.
        
        Returns:
            dict: {
                'mode': GameMode,
                'name': str,
                'speed_multiplier': float,
                'score_multiplier': float,
                'color': tuple (R, G, B)
            }
        """
        speed_mult, score_mult, name, color = self.MODE_CONFIG[self.current_mode]
        return {
            'mode': self.current_mode,
            'name': name,
            'speed_multiplier': speed_mult,
            'score_multiplier': score_mult,
            'color': color
        }
    
    def get_pending_mode(self):
        """
        Get the mode that would be switched to if stable.
        Useful for showing a preview/indicator.
        
        Returns:
            tuple: (GameMode, frames_until_switch)
        """
        frames_remaining = max(0, self.stability_frames - self.pending_count)
        return self.pending_mode, frames_remaining
    
    def force_mode(self, mode):
        """
        Force a specific mode (for testing or special events).
        
        Args:
            mode: GameMode to force
        """
        self.current_mode = mode
        self.pending_mode = mode
        self.pending_count = self.stability_frames
    
    def reset(self):
        """Reset to default state."""
        self.current_mode = GameMode.NORMAL
        self.pending_mode = GameMode.NORMAL
        self.pending_count = 0
        self.mode_history.clear()


# Convenience function for simple usage
def detect_game_mode(finger_count):
    """
    Simple function to detect game mode without class instance.
    Note: Does not include stability (returns mode immediately).
    
    Args:
        finger_count: Number of fingers (0-5)
        
    Returns:
        tuple: (mode_name, speed_multiplier, score_multiplier)
    """
    if finger_count == 0:
        return "SLOW_MO", 0.5, 0.5
    elif finger_count <= 2:
        return "NORMAL", 1.0, 1.0
    elif finger_count == 3:
        return "FAST", 1.5, 1.5
    else:
        return "TURBO", 2.0, 2.0
