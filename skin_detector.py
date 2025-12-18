import cv2
import numpy as np

def detect_skin(hsv_frame):
    if hsv_frame is None:
        return None
        
    # Split the HSV image into separate channels
    h, s, v = cv2.split(hsv_frame)
    
    # A. HSV Check - Balanced thresholds
    # Hue 0-25: Capture full hand (slightly wider for fingers)
    # Saturation 35+: Compromise between hand coverage and wood rejection
    # Value 50+: Good for most lighting
    lower_h, upper_h = 0, 25
    lower_s, upper_s = 35, 255
    lower_v, upper_v = 50, 255
    
    mask_h1 = (h >= lower_h) & (h <= upper_h)
    
    # Range 2: Wrap-around Red range (Hue 165-180)
    lower_h2, upper_h2 = 165, 180
    mask_h2 = (h >= lower_h2) & (h <= upper_h2)
    
    mask_h = mask_h1 | mask_h2
    mask_s = (s >= lower_s) & (s <= upper_s)
    mask_v = (v >= lower_v) & (v <= upper_v)
    
    mask_hsv = mask_h & mask_s & mask_v

    # B. YCrCb Check - Slightly wider for better finger detection
    bgr_temp = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    b, g, r = cv2.split(bgr_temp.astype(np.float32))
    
    y_channel = 0.299 * r + 0.587 * g + 0.114 * b
    cr_channel = (r - y_channel) * 0.713 + 128
    cb_channel = (b - y_channel) * 0.564 + 128
    
    # Balanced ranges for skin
    # Cr: [135, 170] - wider to catch finger tips
    # Cb: [85, 120] - wider
    mask_y = y_channel > 70
    mask_cr = (cr_channel >= 135) & (cr_channel <= 170)
    mask_cb = (cb_channel >= 85) & (cb_channel <= 120)
    
    mask_ycrcb = mask_y & mask_cr & mask_cb
    
    # Both checks must pass
    combined_mask = mask_hsv & mask_ycrcb
    
    skin_mask = combined_mask.astype(np.uint8) * 255
    
    # --- Morphology: Balance noise removal vs finger preservation ---
    # OPENING: Remove noise but preserve fingers (smaller kernel)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # CLOSING: Connect finger parts (moderate)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Final smoothing
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    _, skin_mask = cv2.threshold(skin_mask, 127, 255, cv2.THRESH_BINARY)
    
    return skin_mask
