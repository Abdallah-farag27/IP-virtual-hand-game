import cv2
import numpy as np
import time
import math
from collections import deque

from preprocess import preprocess_frame
from skin_detector import detect_skin
from hand_contour import find_hand_contour


class TemporalSmoother:
    """
    Smooths detection results over time to reduce jitter.
    Uses moving average for position and mode for finger count.
    """
    
    def __init__(self, history_size=7):
        self.history_size = history_size
        self.bbox_history = deque(maxlen=history_size)
        self.finger_history = deque(maxlen=history_size)
        self.lost_frames = 0
        self.max_lost_frames = 5  # After this many frames without detection, reset
    
    def update(self, bbox, finger_count):
        """
        Add new detection to history and return smoothed values.
        
        Args:
            bbox: (x, y, w, h) or None if no detection
            finger_count: int 0-5
            
        Returns:
            tuple: (smoothed_bbox, smoothed_finger_count)
        """
        if bbox is not None:
            self.bbox_history.append(bbox)
            self.finger_history.append(finger_count)
            self.lost_frames = 0
        else:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.bbox_history.clear()
                self.finger_history.clear()
                return None, 0
        
        if len(self.bbox_history) == 0:
            return None, 0
        
        # Smooth bbox using moving average
        avg_x = int(sum(b[0] for b in self.bbox_history) / len(self.bbox_history))
        avg_y = int(sum(b[1] for b in self.bbox_history) / len(self.bbox_history))
        avg_w = int(sum(b[2] for b in self.bbox_history) / len(self.bbox_history))
        avg_h = int(sum(b[3] for b in self.bbox_history) / len(self.bbox_history))
        smoothed_bbox = (avg_x, avg_y, avg_w, avg_h)
        
        # Smooth finger count using mode (most frequent value)
        counts = list(self.finger_history)
        smoothed_fingers = max(set(counts), key=counts.count)
        
        return smoothed_bbox, smoothed_fingers
    
    def reset(self):
        self.bbox_history.clear()
        self.finger_history.clear()
        self.lost_frames = 0

def score_contour_as_hand(contour, bbox, frame_height=720):
    """
    Score a contour on how "hand-like" it is vs "face-like".
    Higher score = more likely to be a hand.
    
    Hands have:
    - Lower circularity (irregular shape with fingers)
    - Convexity defects (valleys between fingers)
    - Moderate solidity (gaps between fingers)
    - Usually in lower half of frame
    - Often wider than tall (horizontal orientation)
    
    Faces have:
    - Higher circularity (oval/round)
    - Few/no significant convexity defects
    - High solidity (solid blob)
    - Usually in upper half of frame
    - Taller than wide (vertical oval)
    """
    x, y, w, h = bbox
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0 or area < 1000:
        return -999  # Very low score to reject
    
    # 1. Circularity: 4π * Area / Perimeter²
    # Perfect circle = 1.0, hand with fingers ≈ 0.3-0.6
    circularity = (4 * math.pi * area) / (perimeter ** 2)
    
    # 2. Solidity: Contour Area / Convex Hull Area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # 3. Aspect ratio: width / height
    # Faces are usually taller than wide (ratio < 1)
    # Hands with fingers spread are often wider (ratio > 0.7)
    aspect_ratio = w / h if h > 0 else 1
    
    # 4. Count finger-like defects
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    finger_defects = 0
    
    if len(contour) >= 3 and len(hull_indices) >= 3:
        try:
            defects = cv2.convexityDefects(contour, hull_indices)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    depth = d / 256.0
                    
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    
                    # Calculate angle at the defect point
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    
                    if b * c > 0:
                        cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
                        cos_angle = max(-1, min(1, cos_angle))
                        angle = math.degrees(math.acos(cos_angle))
                        
                        # Finger valleys: angle < 90°, depth > 15% of bbox height
                        # Stricter depth requirement helps reject noisy contours
                        if angle < 90 and depth > h * 0.15:
                            finger_defects += 1
        except cv2.error:
            pass
    
    # ============ SCORING ============
    score = 0
    
    # --- CRITICAL: Must have at least 1 finger-like defect ---
    # This is the KEY differentiator between hands and faces
    # Faces almost never have proper finger-like convexity defects
    if finger_defects == 0:
        return -50  # Reject contours with no finger features
    
    # Finger defects: huge bonus for having finger-like features
    score += finger_defects * 25  # Up to 100 points for 4 defects
    
    # Circularity score: hands are less circular
    if circularity < 0.6:
        score += (0.6 - circularity) * 40  # Up to 24 points
    else:
        score -= 30  # Penalty for being too circular (face-like)
    
    # Solidity: faces are very solid (~0.9+), hands less so
    if 0.5 <= solidity <= 0.85:
        score += 20  # Bonus for hand-like solidity
    elif solidity > 0.88:
        score -= 30  # Strong penalty for face-like solidity
    
    # Aspect ratio: prefer wider contours (hand-like)
    if aspect_ratio > 0.8:
        score += 15  # Wider than tall, likely hand
    elif aspect_ratio < 0.6:
        score -= 20  # Taller than wide, might be face
    
    # Position: hands are usually in lower half of frame
    # Center Y of contour
    center_y = y + h / 2
    if center_y > frame_height * 0.4:
        score += 15  # In lower 60% of frame
    elif center_y < frame_height * 0.25:
        score -= 25  # In top 25%, likely face
    
    return score


def find_best_hand_contour(skin_mask, frame_height=720):
    """
    Find the contour most likely to be a hand (not face).
    Uses scoring based on shape features.
    """
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None, 0
    
    best_contour = None
    best_bbox = None
    best_score = -999
    best_fingers = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 3000:  # Skip tiny blobs
            continue
        
        bbox = cv2.boundingRect(contour)
        score = score_contour_as_hand(contour, bbox, frame_height)
        
        if score > best_score:
            best_score = score
            best_contour = contour
            best_bbox = bbox
    
    if best_contour is None:
        return None, None, None, 0
    
    # Recalculate finger count for the best contour
    hull_indices = cv2.convexHull(best_contour, returnPoints=False)
    finger_count = 0
    x, y, w, h = best_bbox
    
    if len(best_contour) >= 3 and len(hull_indices) >= 3:
        try:
            defects = cv2.convexityDefects(best_contour, hull_indices)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    depth = d / 256.0
                    start = tuple(best_contour[s][0])
                    end = tuple(best_contour[e][0])
                    far = tuple(best_contour[f][0])
                    
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    
                    if b * c > 0:
                        cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
                        cos_angle = max(-1, min(1, cos_angle))
                        angle = math.degrees(math.acos(cos_angle))
                        
                        if angle < 90 and depth > h * 0.12:
                            finger_count += 1
        except cv2.error:
            pass
    
    finger_count = min(finger_count + 1, 5) if finger_count > 0 else 0
    hull = cv2.convexHull(best_contour, returnPoints=True)
    
    return best_contour, best_bbox, hull, finger_count


def main():
    print("Testing Skin Detection + Smart Hand Selection (NO Background Subtraction)")
    print("=" * 60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found!")
        return
    
    print("Camera found! Running live test.")
    print("Press 'q' to quit, 'r' to reset tracking.")
    
    # Initialize temporal smoother for stable detection
    smoother = TemporalSmoother(history_size=7)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break
            
            t1 = time.time()
            
            # --- 1. Preprocessing ---
            width, height = 960, 720
            frame_resized = cv2.resize(frame, (width, height))
            frame_flipped = frame_resized[:, ::-1]  # Mirror for natural interaction
            
            # Apply Gaussian blur to reduce noise
            frame_blurred = cv2.GaussianBlur(frame_flipped, (5, 5), 0)
            
            # Convert to HSV for skin detection
            hsv_frame = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)
            
            # --- 2. Skin Detection (NO background subtraction) ---
            # Background subtraction causes fading when hand is still
            skin_mask = detect_skin(hsv_frame)
            
            t2 = time.time()
            
            # --- 3. Smart Hand Selection ---
            # Pick the contour most likely to be a hand (not face)
            # The scoring system rejects faces based on shape features
            contour, bbox, hull, finger_count = find_best_hand_contour(skin_mask)
            
            # --- 4. Temporal Smoothing ---
            # Smooth position and finger count over time to reduce jitter
            smoothed_bbox, smoothed_fingers = smoother.update(bbox, finger_count)
            
            t3 = time.time()
            
            # --- Visualization ---
            display_h = 360
            display_w = 480
            
            # 1. Original
            orig_small = cv2.resize(frame_flipped, (display_w, display_h))
            
            # 2. Skin Mask
            skin_small = cv2.resize(skin_mask, (display_w, display_h))
            skin_bgr = cv2.cvtColor(skin_small, cv2.COLOR_GRAY2BGR)
            
            # 3. All contours (for debugging)
            all_contours_frame = frame_flipped.copy()
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 3000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    score = score_contour_as_hand(cnt, (x, y, w, h), height)
                    cv2.drawContours(all_contours_frame, [cnt], -1, (0, 255, 255), 1)
                    cv2.putText(all_contours_frame, f"{score:.0f}", (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            all_contours_small = cv2.resize(all_contours_frame, (display_w, display_h))
            
            # 4. Selected Hand (using smoothed values for display)
            hand_frame = frame_flipped.copy()
            if contour is not None and smoothed_bbox is not None:
                cv2.drawContours(hand_frame, [contour], -1, (0, 255, 0), 3)
                cv2.drawContours(hand_frame, [hull], -1, (255, 0, 0), 2)
                x, y, w, h = smoothed_bbox  # Use smoothed bbox for rectangle
                cv2.rectangle(hand_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                # Show smoothed finger count (more stable)
                cv2.putText(hand_frame, f"HAND - Fingers: {smoothed_fingers}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            hand_small = cv2.resize(hand_frame, (display_w, display_h))
            
            # Layout
            top_row = np.hstack((orig_small, skin_bgr))
            bottom_row = np.hstack((all_contours_small, hand_small))
            display = np.vstack((top_row, bottom_row))
            
            # Labels
            cv2.putText(display, "1. Input", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(display, "2. Skin Mask", (490, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(display, "3. All Contours (scores)", (10, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(display, "4. Selected Hand", (490, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            
            fps = 1.0 / (t3 - t1) if (t3 - t1) > 0 else 0
            cv2.putText(display, f"FPS: {fps:.1f}", (850, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            cv2.imshow("Hand Detection (Skin + Motion)", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('r'):
                smoother.reset()
                print("Tracking reset!")
                
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
