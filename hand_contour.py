import cv2
import numpy as np

def find_hand_contour(binary_mask):
    """
    Finds the largest contour in the binary mask, assuming it is the hand.
    Also calculates the bounding box and convex hull.
    
    Args:
        binary_mask (np.ndarray): Black and white image (0 or 255).
        
    Returns:
        tuple: (hand_contour, bounding_box, convex_hull)
               - hand_contour: The list of points making up the hand shape.
               - bounding_box: (x, y, w, h) rectangle around the hand.
               - convex_hull: The polygon wrapping the hand (like a rubber band).
               Returns (None, None, None) if no valid hand is found.
    """
    # 1. Find all contours
    # RETR_EXTERNAL: Only retrieve the outermost contours (don't care about holes inside the hand)
    # CHAIN_APPROX_SIMPLE: Compresses horizontal/vertical segments to save memory
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None
        
    # 2. Filter by Area (Find the largest contour)
    # The hand should be the biggest white object in the mask.
    # This filters out small isolated noise blobs.
    max_area = 0
    largest_contour = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_contour = cnt
            
    # Add a minimum area threshold to avoid detecting random noise as a hand
    # e.g., if the largest thing is just a 5x5 pixel dot, ignore it.
    if max_area < 3000:  # 3000 pixels is roughly a small fist size at 720p
        return None, None, None
        
    # 3. Calculate Bounding Rectangle
    # (x, y) is top-left corner, (w, h) is width and height
    x, y, w, h = cv2.boundingRect(largest_contour)
    bounding_box = (x, y, w, h)
    
    # 4. Calculate Convex Hull
    # The convex hull is the "rubber band" stretched around the object.
    # It connects the outermost points (like fingertips).
    # returnPoints=True gives us the actual coordinate points of the hull.
    convex_hull = cv2.convexHull(largest_contour, returnPoints=True)
    
    return largest_contour, bounding_box, convex_hull
