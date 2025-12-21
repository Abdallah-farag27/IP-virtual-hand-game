import cv2
import numpy as np
import math

class HandDetector:
    """
    Hand detection using classical image processing techniques.
    Step 1: ROI Restriction + Skin Detection
    """
    
    def __init__(self, roi_height_ratio=0.5, min_area=5000, max_area=50000):
        """
        Initialize hand detector.
        
        Args:
            roi_height_ratio: Fraction of frame height to use as ROI (from bottom)
            min_area: Minimum contour area to consider as hand
            max_area: Maximum contour area to consider as hand
        """
        self.roi_height_ratio = roi_height_ratio
        self.min_area = min_area
        self.max_area = max_area
        
        # Skin color ranges in HSV
        # Lower range (reddish skin tones)
        self.lower_skin_1 = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin_1 = np.array([20, 255, 255], dtype=np.uint8)
        
        # Upper range (yellowish skin tones)
        self.lower_skin_2 = np.array([170, 20, 70], dtype=np.uint8)
        self.upper_skin_2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # Temporal smoothing
        self.prev_x = None
        self.smoothing_alpha = 0.3  # Exponential smoothing factor
        
    def get_roi_bounds(self, frame_height):
        """Calculate ROI boundaries (lower portion of frame)."""
        roi_start_y = int(frame_height * (1 - self.roi_height_ratio))
        return roi_start_y, frame_height
    
    def detect_skin(self, frame, use_ycrcb=False):
        """
        Detect skin-colored regions using HSV color space.
        Optionally combines with YCrCb for more robust detection.
        
        Args:
            frame: BGR image
            use_ycrcb: Whether to also use YCrCb color space
            
        Returns:
            Binary mask of skin regions
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply skin color thresholds (HSV)
        mask1 = cv2.inRange(hsv, self.lower_skin_1, self.upper_skin_1)
        mask2 = cv2.inRange(hsv, self.lower_skin_2, self.upper_skin_2)
        
        # Combine HSV masks
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Optionally add YCrCb detection for more robustness
        if use_ycrcb:
            # Convert to YCrCb
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            
            # YCrCb skin color range
            lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
            upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
            
            # Apply YCrCb threshold
            mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
            
            # Combine with HSV mask (AND operation for more precision)
            skin_mask = cv2.bitwise_and(skin_mask, mask_ycrcb)
        
        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Remove small noise
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Fill holes
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Dilate to make hand more prominent
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
        
        return skin_mask
    
    def find_hand_contour(self, mask):
        """
        Find the largest valid hand contour in the mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Largest contour or None if no valid contour found
        """
        # Find all contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by area
        valid_contours = [c for c in contours 
                         if self.min_area <= cv2.contourArea(c) <= self.max_area]
        
        if not valid_contours:
            return None
        
        # Return largest valid contour
        return max(valid_contours, key=cv2.contourArea)
    
    def get_contour_centroid(self, contour):
        """
        Calculate centroid of contour.
        
        Args:
            contour: OpenCV contour
            
        Returns:
            (x, y) tuple of centroid coordinates
        """
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    
    def get_palm_center_distance_transform(self, mask, contour):
        """
        Find palm center using distance transform.
        More stable than centroid when fingers are moving.
        
        Args:
            mask: Binary mask of hand
            contour: Hand contour
            
        Returns:
            (x, y) tuple of palm center coordinates, or None
            distance_transform: Distance transform image for visualization
        """
        # Create mask for just this contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        
        # Apply distance transform
        dist_transform = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 5)
        
        # Find the point with maximum distance (palm center)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
        
        if max_val > 0:
            return max_loc, dist_transform
        
        return None, dist_transform

    
    def apply_temporal_smoothing(self, x):
        """
        Apply exponential smoothing to x-coordinate.
        
        Args:
            x: Current x-coordinate
            
        Returns:
            Smoothed x-coordinate
        """
        if self.prev_x is None:
            self.prev_x = x
            return x
        
        # Exponential moving average
        smoothed_x = int(self.smoothing_alpha * x + (1 - self.smoothing_alpha) * self.prev_x)
        self.prev_x = smoothed_x
        return smoothed_x
    
    def detect_hand(self, frame, use_smoothing=True):
        """
        Detect hand in frame using ROI restriction and skin detection.
        
        Args:
            frame: BGR image
            use_smoothing: Whether to apply temporal smoothing
            
        Returns:
            Dictionary with hand information:
            {
                'x': x-coordinate of hand center,
                'y': y-coordinate of hand center,
                'contour': hand contour,
                'roi_start_y': ROI start y-coordinate,
                'detected': True if hand found
            }
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Get ROI bounds
        roi_start_y, roi_end_y = self.get_roi_bounds(frame_height)
        
        # Extract ROI (lower portion of frame)
        roi = frame[roi_start_y:roi_end_y, :]
        
        # Detect skin in ROI
        skin_mask = self.detect_skin(roi)
        
        # Find hand contour
        hand_contour = self.find_hand_contour(skin_mask)
        
        result = {
            'detected': False,
            'x': None,
            'y': None,
            'contour': None,
            'roi_start_y': roi_start_y,
            'skin_mask': skin_mask
        }
        
        if hand_contour is not None:
            # Get centroid in ROI coordinates
            centroid = self.get_contour_centroid(hand_contour)
            
            if centroid is not None:
                cx, cy = centroid
                
                # Convert to full frame coordinates
                full_frame_y = cy + roi_start_y
                
                # Apply temporal smoothing to x-coordinate
                if use_smoothing:
                    smoothed_x = self.apply_temporal_smoothing(cx)
                else:
                    smoothed_x = cx
                
                result['detected'] = True
                result['x'] = smoothed_x
                result['y'] = full_frame_y
                result['contour'] = hand_contour
                result['raw_x'] = cx  # Store raw x for comparison
        else:
            # No hand detected, use previous position if available
            if self.prev_x is not None and use_smoothing:
                result['x'] = self.prev_x
                result['y'] = frame_height - 100  # Default y position
        
        return result
    
    def detect_hand_with_palm_center(self, frame, use_smoothing=True):
        """
        Detect hand using distance transform for palm center detection.
        More stable than centroid-based detection.
        
        Args:
            frame: BGR image
            use_smoothing: Whether to apply temporal smoothing
            
        Returns:
            Dictionary with hand information including distance transform
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Get ROI bounds
        roi_start_y, roi_end_y = self.get_roi_bounds(frame_height)
        
        # Extract ROI (lower portion of frame)
        roi = frame[roi_start_y:roi_end_y, :]
        
        # Detect skin in ROI
        skin_mask = self.detect_skin(roi)
        
        # Find hand contour
        hand_contour = self.find_hand_contour(skin_mask)
        
        result = {
            'detected': False,
            'x': None,
            'y': None,
            'contour': None,
            'roi_start_y': roi_start_y,
            'skin_mask': skin_mask,
            'distance_transform': None,
            'palm_radius': 0
        }
        
        if hand_contour is not None:
            # Get palm center using distance transform
            palm_center, dist_transform = self.get_palm_center_distance_transform(skin_mask, hand_contour)
            
            result['distance_transform'] = dist_transform
            
            if palm_center is not None:
                cx, cy = palm_center
                
                # Get palm radius (max distance value)
                palm_radius = dist_transform[cy, cx]
                
                # Convert to full frame coordinates
                full_frame_y = cy + roi_start_y
                
                # Apply temporal smoothing to x-coordinate
                if use_smoothing:
                    smoothed_x = self.apply_temporal_smoothing(cx)
                else:
                    smoothed_x = cx
                
                result['detected'] = True
                result['x'] = smoothed_x
                result['y'] = full_frame_y
                result['contour'] = hand_contour
                result['raw_x'] = cx
                result['palm_radius'] = palm_radius
        else:
            # No hand detected, use previous position if available
            if self.prev_x is not None and use_smoothing:
                result['x'] = self.prev_x
                result['y'] = frame_height - 100
        
        return result

    def count_fingers(self, contour):
        """
        Count fingers using convexity defects.
        
        Args:
            contour: Hand contour
            
        Returns:
            Number of fingers extended (0-5)
        """
        if contour is None:
            return 0
            
        # Hull with indices
        hull = cv2.convexHull(contour, returnPoints=False)
        
        # Need at least 3 points for defects
        if len(hull) < 3:
            return 0
            
        # Get defects
        # defects[i] = [start_index, end_index, farthest_pt_index, fixpt_depth]
        defects = cv2.convexityDefects(contour, hull)
        
        if defects is None:
            return 0
            
        count = 0
        
        # Use simple logic: number of fingers = number of deep defects + 1
        # But we need to filter defects carefully
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Depth of the defect (distance from hull to contour)
            # defects returns fixed point depth (scaled by 256)
            depth = d / 256.0
            
            # Only consider deep defects (gaps between fingers)
            if depth > 30:  # Increased threshold to reduce flickering (was 20)
                # Calculate sides of the triangle formed by start, end, far
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                
                # Apply cosine rule to find angle at the 'far' point (the gap)
                # We want acute angles (between fingers)
                if b * c > 0: # Avoid division by zero
                    angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
                    angle = angle * 180 / math.pi
                    
                    # Angle between fingers is usually less than 90 degrees
                    if angle < 90:
                        count += 1
                        
        # If we found defects, it means we have fingers up.
        # N defects usually means N+1 fingers.
        if count > 0:
            return count + 1
        
        # Special case: 0 defects usually means 0 fingers (fist) or 1 finger.
        # Distinguish using shape properties.
        if count == 0:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / float(w)
            
            area = cv2.contourArea(contour)
            hull_points = cv2.convexHull(contour, returnPoints=True)
            hull_area = cv2.contourArea(hull_points)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Heuristic: 
            # 1 Finger is usually "tall" (High aspect ratio) AND/OR has lower solidity (not a perfect blob) due to the stick shape.
            # Fist is usually a "blob" (Square-ish aspect ratio, High solidity).
            
            # If height is significantly larger than width (tall), typically 1 finger up.
            # OR if solidity is low (meaning hull is much larger than contour), it's likely a finger sticking out.
            if aspect_ratio > 1.3 or solidity < 0.88:
                return 1
                
            return 0
        
        return 0

    
    def reset(self):
        """Reset temporal smoothing state."""
        self.prev_x = None
