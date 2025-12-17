import cv2
import numpy as np

def detect_skin(hsv_frame):
    if hsv_frame is None:
        return None
        
    # Split the HSV image into separate channels to process them individually
    h, s, v = cv2.split(hsv_frame)
    
    # A. HSV Check
    # We define TWO ranges for Hue because red appears at both ends of the color wheel.
    
    # Range 1: Standard Red-Yellow range (Hue 0-50)
    # S > 15 avoids whites/grays. V > 40 avoids absolute black shadows.
    lower_h, upper_h = 0, 50
    lower_s, upper_s = 15, 255
    lower_v, upper_v = 40, 255
    
    # Create mask for Range 1
    # Check if pixel values are within the defined bounds
    mask_h1 = (h >= lower_h) & (h <= upper_h)
    
    # Range 2: Wrap-around Red range (Hue 170-180)
    # This captures deep reds that are near the end of the spectrum.
    lower_h2, upper_h2 = 170, 180
    mask_h2 = (h >= lower_h2) & (h <= upper_h2)
    
    # Combine Hue masks: Valid if in Range 1 OR Range 2
    mask_h = mask_h1 | mask_h2
    
    # Create S and V masks
    mask_s = (s >= lower_s) & (s <= upper_s)
    mask_v = (v >= lower_v) & (v <= upper_v)
    
    # Final HSV Mask: Must meet Hue AND Saturation AND Value criteria
    mask_hsv = mask_h & mask_s & mask_v

    # B. YCrCb Check (Manual Implementation)
    # YCrCb is another color space often used for skin detection.
    # We convert the image manually using the standard formulas.
    
    # First, convert HSV back to BGR so we can calculate YCrCb
    # (Using OpenCV for this intermediate step for efficiency)
    bgr_temp = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    
    # Split BGR channels
    b, g, r = cv2.split(bgr_temp.astype(np.float32))
    
    # Calculate Y, Cr, Cb
    # Y = Luma (Brightness), Cr = Red-difference, Cb = Blue-difference
    y_channel = 0.299 * r + 0.587 * g + 0.114 * b
    cr_channel = (r - y_channel) * 0.713 + 128
    cb_channel = (b - y_channel) * 0.564 + 128
    
    # Check if pixels fall within typical skin ranges in YCrCb
    # Cr: [133, 173], Cb: [77, 127]
    mask_y = y_channel > 80
    mask_cr = (cr_channel >= 133) & (cr_channel <= 173)
    mask_cb = (cb_channel >= 77) & (cb_channel <= 127)
    
    # Combine YCrCb conditions
    mask_ycrcb = mask_y & mask_cr & mask_cb
    
    # --- Combine Both Methods ---
    # Intersection: Pixel must be skin in BOTH HSV and YCrCb models.
    # This removes almost all background noise that might pass one check but not the other.
    combined_mask = mask_hsv & mask_ycrcb
    
    # Convert Boolean mask (True/False) to Image mask (255/0)
    skin_mask = combined_mask.astype(np.uint8) * 255
    
    # --- 2. Morphological Operations (Noise Cancellation) ---
    
    # Step A: OPENING (Erosion -> Dilation)
    # Removes small white dots (background noise)
    # We use an elliptical kernel shape
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # Step B: CLOSING (Dilation -> Erosion)
    # Fills small black holes inside the white hand region
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Final smoothing with Gaussian Blur to soften the edges
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    
    # Threshold again to ensure strict black/white result
    _, skin_mask = cv2.threshold(skin_mask, 127, 255, cv2.THRESH_BINARY)
    
    return skin_mask
