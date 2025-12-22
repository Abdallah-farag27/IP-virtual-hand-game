import cv2
import mediapipe as mp
import numpy as np

class MediaPipeDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Finger tip IDs: Thumb, Index, Middle, Ring, Pinky
        self.tip_ids = [4, 8, 12, 16, 20]
        
    def detect_hand(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        frame_height, frame_width = frame.shape[:2]
        
        roi_start_y = 0 
        
        hand_info = {
            'detected': False,
            'x': None,
            'y': None,
            'roi_start_y': roi_start_y,
            'contour': None,
            'finger_count': 0,
            'landmarks': None,
            'type': 'mediapipe'
        }
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Calculate center (simple average of all landmarks or specific palm point)
            # Use Index Finger MCP (5) and Pinky MCP (17) and Wrist (0) for palm center?
            # Or just use the Wrist (0) or Middle Finger MCP (9)
            
            # Let's use landmark 9 (MIDDLE_FINGER_MCP) as a stable center proxy
            lm_x = hand_landmarks.landmark[9].x
            lm_y = hand_landmarks.landmark[9].y
            
            cx, cy = int(lm_x * frame_width), int(lm_y * frame_height)
            
            hand_info['detected'] = True
            hand_info['x'] = cx
            hand_info['y'] = cy
            hand_info['landmarks'] = hand_landmarks
            
            # Count fingers
            hand_info['finger_count'] = self.count_fingers(hand_landmarks)
            
        return hand_info

    def detect_hand_with_palm_center(self, frame, use_smoothing=False):
        # Wrapper to match API, MediaPipe doesn't really need a separate "palm center" specific function
        # as landmarks give us precise positions anyway.
        return self.detect_hand(frame, use_smoothing)

    def count_fingers(self, hand_landmarks):
        fingers = []
    
        tx, ty = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y
        ipx, ipy = hand_landmarks.landmark[3].x, hand_landmarks.landmark[3].y
        
        if abs(tx - hand_landmarks.landmark[17].x) > abs(ipx - hand_landmarks.landmark[17].x):
             fingers.append(1)
        else:
             fingers.append(0)

        for id in range(1, 5):
            tip_id = self.tip_ids[id] # 8, 12, 16, 20
            # Compare tip y to pip y (landmark - 2)
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers.count(1)

    def get_roi_bounds(self, frame_height):
        # MediaPipe searches full frame usually, but we can return same dummies
        return 0, frame_height

    def reset(self):
        pass
