"""
Finger Counter Module - Uses MediaPipe Hand Landmarks
Part of: Fararg & Tony's contribution

Counts the number of extended fingers using MediaPipe's 21 hand landmarks.
Returns a stable finger count (0-5) with temporal smoothing.
"""

from collections import deque


class FingerCounter:
    """
    Counts fingers using MediaPipe hand landmarks.
    
    MediaPipe provides 21 landmarks per hand:
    - 0: Wrist
    - 1-4: Thumb (CMC, MCP, IP, TIP)
    - 5-8: Index finger
    - 9-12: Middle finger
    - 13-16: Ring finger
    - 17-20: Pinky finger
    
    A finger is "up" if its TIP is above its PIP joint (lower y value).
    Thumb needs special handling based on x-position.
    """
    
    # Landmark indices
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
    
    def __init__(self, smoothing_window=5):
        """
        Initialize finger counter with temporal smoothing.
        
        Args:
            smoothing_window: Number of frames to average for stability
        """
        self.smoothing_window = smoothing_window
        self.history = deque(maxlen=smoothing_window)
    
    def count_fingers(self, hand_landmarks, handedness="Right"):
        """
        Count extended fingers from MediaPipe hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            handedness: "Left" or "Right" hand (affects thumb detection)
            
        Returns:
            int: Number of fingers up (0-5), smoothed over time
        """
        if hand_landmarks is None:
            return self._get_smoothed_count(0)
        
        count = 0
        
        # Check each finger
        # Thumb - check x position relative to MCP
        count += self._is_thumb_up(hand_landmarks, handedness)
        
        # Index finger - check if TIP is above PIP
        count += self._is_finger_up(hand_landmarks, self.INDEX_TIP, self.INDEX_PIP)
        
        # Middle finger
        count += self._is_finger_up(hand_landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP)
        
        # Ring finger
        count += self._is_finger_up(hand_landmarks, self.RING_TIP, self.RING_PIP)
        
        # Pinky finger
        count += self._is_finger_up(hand_landmarks, self.PINKY_TIP, self.PINKY_PIP)
        
        return self._get_smoothed_count(count)
    
    def _is_thumb_up(self, landmarks, handedness):
        """
        Check if thumb is extended.
        
        For right hand: thumb TIP should be LEFT of MCP (lower x in mirrored view)
        For left hand: thumb TIP should be RIGHT of MCP (higher x in mirrored view)
        
        Args:
            landmarks: MediaPipe hand landmarks
            handedness: "Left" or "Right"
            
        Returns:
            int: 1 if thumb is up, 0 otherwise
        """
        thumb_tip = landmarks.landmark[self.THUMB_TIP]
        thumb_mcp = landmarks.landmark[self.THUMB_MCP]
        
        # Threshold for thumb extension
        threshold = 0.04
        
        if handedness == "Right":
            # In mirrored view, right thumb extends left (lower x)
            if thumb_tip.x < thumb_mcp.x - threshold:
                return 1
        else:
            # In mirrored view, left thumb extends right (higher x)
            if thumb_tip.x > thumb_mcp.x + threshold:
                return 1
        
        return 0
    
    def _is_finger_up(self, landmarks, tip_idx, pip_idx):
        """
        Check if a finger is extended by comparing TIP and PIP y positions.
        
        A finger is "up" if TIP is above PIP (lower y value in image coordinates).
        
        Args:
            landmarks: MediaPipe hand landmarks
            tip_idx: Index of fingertip landmark
            pip_idx: Index of PIP joint landmark
            
        Returns:
            int: 1 if finger is up, 0 otherwise
        """
        tip = landmarks.landmark[tip_idx]
        pip = landmarks.landmark[pip_idx]
        
        # Finger is up if tip is above pip (lower y value)
        # Add small threshold to avoid noise
        if tip.y < pip.y - 0.02:
            return 1
        return 0
    
    def _get_smoothed_count(self, current_count):
        """
        Apply temporal smoothing to finger count using mode (most common value).
        
        Args:
            current_count: Raw finger count for current frame
            
        Returns:
            int: Smoothed finger count
        """
        self.history.append(current_count)
        
        if len(self.history) == 0:
            return current_count
        
        # Return mode (most common value) for stability
        counts = list(self.history)
        return max(set(counts), key=counts.count)
    
    def get_finger_states(self, hand_landmarks, handedness="Right"):
        """
        Get individual finger states (which specific fingers are up).
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            handedness: "Left" or "Right"
            
        Returns:
            dict: {'thumb': bool, 'index': bool, 'middle': bool, 'ring': bool, 'pinky': bool}
        """
        if hand_landmarks is None:
            return {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False}
        
        return {
            'thumb': self._is_thumb_up(hand_landmarks, handedness) == 1,
            'index': self._is_finger_up(hand_landmarks, self.INDEX_TIP, self.INDEX_PIP) == 1,
            'middle': self._is_finger_up(hand_landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP) == 1,
            'ring': self._is_finger_up(hand_landmarks, self.RING_TIP, self.RING_PIP) == 1,
            'pinky': self._is_finger_up(hand_landmarks, self.PINKY_TIP, self.PINKY_PIP) == 1
        }
    
    def reset(self):
        """Clear the smoothing history."""
        self.history.clear()


# Convenience function for simple usage
def count_fingers(hand_landmarks, handedness="Right"):
    """
    Simple function to count fingers without creating a class instance.
    Note: Does not include temporal smoothing.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        handedness: "Left" or "Right"
        
    Returns:
        int: Number of fingers up (0-5)
    """
    counter = FingerCounter(smoothing_window=1)
    return counter.count_fingers(hand_landmarks, handedness)
