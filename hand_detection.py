"""
Hand Detection Module

Implements hand detection using skin-color segmentation with HSV thresholding
and morphological operations, following classic image processing techniques.
"""

import cv2
import numpy as np


def preprocess(frame):
    """
    Removes noise and prepares frame for segmentation.
    
    Args:
        frame: BGR image from webcam
        
    Returns:
        HSV-converted frame with noise reduced via Gaussian blur
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    
    # Convert to HSV color space for skin detection
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    return hsv


def segment_hand(hsv_frame):
    """
    Detects hand region using skin-color segmentation in HSV space.
    
    Args:
        hsv_frame: HSV-converted frame from preprocess()
        
    Returns:
        Binary mask where white pixels indicate detected skin/hand region
    """
    # Define skin color range in HSV
    # These values work well for a variety of skin tones under typical lighting
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    
    # Second range for reddish skin tones (hue wraps around)
    lower_skin2 = np.array([160, 30, 60], dtype=np.uint8)
    upper_skin2 = np.array([180, 150, 255], dtype=np.uint8)
    
    # Create masks for both skin color ranges
    mask1 = cv2.inRange(hsv_frame, lower_skin, upper_skin)
    mask2 = cv2.inRange(hsv_frame, lower_skin2, upper_skin2)
    
    # Combine masks
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Morphological cleanup to remove noise and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Erode to remove small noise
    mask = cv2.erode(mask, kernel, iterations=2)
    
    # Dilate to restore hand region and fill small gaps
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    return mask


def get_hand_centroid(mask):
    """
    Finds the largest contour in the mask and computes its centroid.
    
    Args:
        mask: Binary mask from segment_hand()
        
    Returns:
        Dictionary {'x': int, 'y': int} with centroid coordinates,
        or None if no hand is detected
    """
    # Find all contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour (assumed to be the hand)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Filter out small contours (noise)
    if cv2.contourArea(largest_contour) < 3000:
        return None
    
    # Calculate moments to find centroid
    moments = cv2.moments(largest_contour)
    
    if moments['m00'] == 0:
        return None
    
    # Calculate centroid coordinates
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    
    return {'x': cx, 'y': cy}


def draw_debug(frame, mask, centroid):
    """
    Draws debug visualization on the frame (optional helper function).
    
    Args:
        frame: Original BGR frame to draw on
        mask: Binary mask from segmentation
        centroid: Centroid dict or None
    """
    # Draw mask overlay (semi-transparent green where hand is detected)
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_colored[:, :, 0] = 0  # Remove blue
    mask_colored[:, :, 2] = 0  # Remove red (keep green)
    frame_with_mask = cv2.addWeighted(frame, 1, mask_colored, 0.3, 0)
    
    # Draw centroid if detected
    if centroid:
        cv2.circle(frame_with_mask, (centroid['x'], centroid['y']), 15, (0, 255, 255), -1)
        cv2.circle(frame_with_mask, (centroid['x'], centroid['y']), 15, (0, 0, 0), 2)
    
    return frame_with_mask
