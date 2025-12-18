import cv2
import numpy as np
import math

def count_fingers_by_defects(contour, defects, bbox):
    """
    Count the number of fingers using convexity defects.
    
    Convexity defects are the "valleys" between fingers when you compare
    the hand contour to its convex hull (rubber band around the hand).
    
    Args:
        contour: The hand contour points
        defects: Convexity defects from cv2.convexityDefects()
        bbox: Bounding box (x, y, w, h) of the contour
        
    Returns:
        int: Estimated number of fingers (0-5)
    """
    if defects is None:
        return 0
    
    x, y, w, h = bbox
    finger_count = 0
    
    for i in range(defects.shape[0]):
        # Each defect has 4 values:
        # start_index: Starting point of defect on contour
        # end_index: Ending point of defect on contour
        # farthest_index: Deepest point of the defect (valley between fingers)
        # distance: Distance from the farthest point to the hull (depth of valley)
        s, e, f, d = defects[i, 0]
        
        start = tuple(contour[s][0])    # Fingertip 1
        end = tuple(contour[e][0])      # Fingertip 2  
        far = tuple(contour[f][0])      # Valley between fingers
        
        # Calculate the lengths of the triangle formed by start, end, far
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        
        # Use law of cosines to find angle at the "far" point (valley)
        # cos(angle) = (b² + c² - a²) / (2bc)
        # A finger valley typically has an angle less than 90 degrees
        s_angle = (b**2 + c**2 - a**2)
        divisor = 2 * b * c
        
        if divisor == 0:
            continue
            
        cos_angle = s_angle / divisor
        # Clamp to valid range to avoid math domain errors
        cos_angle = max(-1, min(1, cos_angle))
        angle = math.acos(cos_angle)
        angle_deg = math.degrees(angle)
        
        # Convert depth from fixed-point to float (OpenCV stores as fixed-point)
        depth = d / 256.0
        
        # A valid finger valley should have:
        # 1. Angle less than 90 degrees (fingers spread apart)
        # 2. Depth greater than ~20% of the bounding box height
        min_depth = h * 0.15  # At least 15% of hand height
        
        if angle_deg < 90 and depth > min_depth:
            finger_count += 1
    
    # The number of defects between fingers = fingers - 1
    # So if we found 4 defects, that's 5 fingers
    # But we cap at 5 to be safe
    return min(finger_count + 1, 5) if finger_count > 0 else 0


def is_valid_hand_shape(contour, bbox):
    """
    Validates if a contour looks like a hand based on:
    1. Aspect ratio (hands are roughly square-ish, not super elongated)
    2. Solidity (ratio of contour area to convex hull area)
    3. Convexity defects (fingers create valleys in the hull)
    
    Args:
        contour: The contour to validate
        bbox: Bounding box (x, y, w, h)
        
    Returns:
        tuple: (is_valid, finger_count, defects)
    """
    x, y, w, h = bbox
    
    # 1. Check aspect ratio (width/height should be reasonable)
    # A hand is usually between 0.5 and 2.0 aspect ratio
    aspect_ratio = w / h if h > 0 else 0
    if aspect_ratio < 0.3 or aspect_ratio > 3.0:
        return False, 0, None
    
    # 2. Calculate solidity
    # Solidity = Contour Area / Convex Hull Area
    # A hand (with finger gaps) typically has solidity between 0.6 and 0.95
    contour_area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area == 0:
        return False, 0, None
        
    solidity = contour_area / hull_area
    
    # If solidity is too low, it might be a weird shape
    # If too high, there are no finger gaps (might be fist or other body part)
    # Note: A closed fist has high solidity (~0.9+), which is still a valid hand
    if solidity < 0.4:
        return False, 0, None
    
    # 3. Calculate convexity defects to count potential fingers
    # We need the hull indices (not points) for defect calculation
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    
    # Need at least 3 points to calculate defects
    if len(contour) < 3 or len(hull_indices) < 3:
        return False, 0, None
    
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return False, 0, None
    
    finger_count = count_fingers_by_defects(contour, defects, bbox)
    
    # A valid hand should potentially show 0-5 fingers
    # Even a fist (0 visible finger gaps) is valid
    # We consider it "hand-like" if it has reasonable solidity
    # OR if it shows finger-like defects
    is_hand_like = (0.5 <= solidity <= 0.95) or finger_count >= 1
    
    return is_hand_like, finger_count, defects


def find_hand_contour(binary_mask, validate_shape=True):
    """
    Finds the largest contour in the binary mask that looks like a hand.
    Uses convexity defect analysis to validate hand shape.
    
    Args:
        binary_mask (np.ndarray): Black and white image (0 or 255).
        validate_shape (bool): If True, validates contour looks like a hand.
        
    Returns:
        tuple: (hand_contour, bounding_box, convex_hull, finger_count)
               - hand_contour: The list of points making up the hand shape.
               - bounding_box: (x, y, w, h) rectangle around the hand.
               - convex_hull: The polygon wrapping the hand (like a rubber band).
               - finger_count: Estimated number of visible fingers (0-5).
               Returns (None, None, None, 0) if no valid hand is found.
    """
    # 1. Find all contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None, 0
    
    # 2. Sort contours by area (largest first)
    # We'll check the largest ones for hand-like features
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Minimum area threshold
    MIN_AREA = 3000
    
    # 3. Check top contours for hand-like shape
    for contour in sorted_contours[:5]:  # Check up to 5 largest contours
        area = cv2.contourArea(contour)
        
        if area < MIN_AREA:
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        bbox = (x, y, w, h)
        
        if validate_shape:
            # Validate this looks like a hand
            is_valid, finger_count, defects = is_valid_hand_shape(contour, bbox)
            
            if not is_valid:
                continue
        else:
            finger_count = 0
        
        # Found a valid hand!
        convex_hull = cv2.convexHull(contour, returnPoints=True)
        return contour, bbox, convex_hull, finger_count
    
    # No valid hand found
    return None, None, None, 0


# Backwards compatible wrapper for existing code that expects 3 return values
def find_hand_contour_simple(binary_mask):
    """
    Simple wrapper that returns 3 values for backwards compatibility.
    """
    contour, bbox, hull, _ = find_hand_contour(binary_mask, validate_shape=True)
    return contour, bbox, hull
