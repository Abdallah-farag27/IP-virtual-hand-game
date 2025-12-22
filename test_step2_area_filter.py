import cv2
import numpy as np
from src.modules.hand_detector import HandDetector

class TestStep2AreaFilter:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize hand detector
        self.detector = HandDetector(
            roi_height_ratio=0.5,
            min_area=5000,
            max_area=50000
        )
        
        # FPS calculation
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = cv2.getTickCount()
    
    def calculate_fps(self):
        self.frame_count += 1
        if self.frame_count >= 10:
            current_time = cv2.getTickCount()
            elapsed = (current_time - self.fps_start_time) / cv2.getTickFrequency()
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def get_all_contours_info(self, frame):
        frame_height, frame_width = frame.shape[:2]
        roi_start_y, roi_end_y = self.detector.get_roi_bounds(frame_height)
        roi = frame[roi_start_y:roi_end_y, :]
        
        # Detect skin
        skin_mask = self.detector.detect_skin(roi)
        
        # Find all contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Classify contours
        too_small = []
        too_large = []
        valid = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.detector.min_area:
                too_small.append((contour, area))
            elif area > self.detector.max_area:
                too_large.append((contour, area))
            else:
                valid.append((contour, area))
        
        # Find largest valid contour
        largest_valid = None
        if valid:
            largest_valid = max(valid, key=lambda x: x[1])
        
        return {
            'too_small': too_small,
            'too_large': too_large,
            'valid': valid,
            'largest_valid': largest_valid,
            'roi_start_y': roi_start_y,
            'skin_mask': skin_mask
        }
    
    def create_visualization(self, frame, contour_info, hand_info):
        frame_height, frame_width = frame.shape[:2]
        roi_y = contour_info['roi_start_y']
        
        # Panel 1: All contours with color coding
        panel1 = frame.copy()
        
        # Draw ROI boundary
        cv2.line(panel1, (0, roi_y), (frame_width, roi_y), (0, 255, 255), 2)
        
        # Draw too small contours (red)
        for contour, area in contour_info['too_small']:
            contour_offset = contour.copy()
            contour_offset[:, :, 1] += roi_y
            cv2.drawContours(panel1, [contour_offset], -1, (0, 0, 255), 2)
        
        # Draw too large contours (blue)
        for contour, area in contour_info['too_large']:
            contour_offset = contour.copy()
            contour_offset[:, :, 1] += roi_y
            cv2.drawContours(panel1, [contour_offset], -1, (255, 0, 0), 2)
        
        # Draw valid contours (yellow)
        for contour, area in contour_info['valid']:
            contour_offset = contour.copy()
            contour_offset[:, :, 1] += roi_y
            cv2.drawContours(panel1, [contour_offset], -1, (0, 255, 255), 2)
        
        # Draw largest valid contour (green, thick)
        if contour_info['largest_valid']:
            contour, area = contour_info['largest_valid']
            contour_offset = contour.copy()
            contour_offset[:, :, 1] += roi_y
            cv2.drawContours(panel1, [contour_offset], -1, (0, 255, 0), 4)
        
        cv2.putText(panel1, "1. All Contours (Color Coded)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 2: Contour statistics
        panel2 = np.zeros_like(frame)
        
        y_pos = 80
        line_height = 35
        
        cv2.putText(panel2, "CONTOUR STATISTICS", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        y_pos += line_height + 20
        
        # Too small
        cv2.putText(panel2, f"Too Small (< {self.detector.min_area}):", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += line_height
        cv2.putText(panel2, f"  Count: {len(contour_info['too_small'])}", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_height + 10
        
        # Too large
        cv2.putText(panel2, f"Too Large (> {self.detector.max_area}):", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        y_pos += line_height
        cv2.putText(panel2, f"  Count: {len(contour_info['too_large'])}", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_height + 10
        
        # Valid
        cv2.putText(panel2, "Valid Contours:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += line_height
        cv2.putText(panel2, f"  Count: {len(contour_info['valid'])}", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_height
        
        # Show areas of valid contours
        if contour_info['valid']:
            y_pos += 10
            cv2.putText(panel2, "Valid Contour Areas:", (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            y_pos += line_height - 5
            
            for i, (contour, area) in enumerate(contour_info['valid'][:5]):  # Show max 5
                cv2.putText(panel2, f"  #{i+1}: {int(area)}", (60, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += line_height - 10
        
        # Largest valid
        y_pos += 20
        if contour_info['largest_valid']:
            contour, area = contour_info['largest_valid']
            cv2.putText(panel2, "Selected Hand:", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += line_height
            cv2.putText(panel2, f"  Area: {int(area)}", (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(panel2, "No Valid Hand Found", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Legend
        legend_y = frame_height - 180
        cv2.putText(panel2, "COLOR LEGEND:", (20, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        legend_y += line_height
        
        cv2.rectangle(panel2, (20, legend_y - 20), (50, legend_y), (0, 0, 255), -1)
        cv2.putText(panel2, "Too Small", (60, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        legend_y += line_height - 5
        
        cv2.rectangle(panel2, (20, legend_y - 20), (50, legend_y), (255, 0, 0), -1)
        cv2.putText(panel2, "Too Large", (60, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        legend_y += line_height - 5
        
        cv2.rectangle(panel2, (20, legend_y - 20), (50, legend_y), (0, 255, 255), -1)
        cv2.putText(panel2, "Valid", (60, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        legend_y += line_height - 5
        
        cv2.rectangle(panel2, (20, legend_y - 20), (50, legend_y), (0, 255, 0), -1)
        cv2.putText(panel2, "Selected", (60, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(panel2, "2. Statistics & Legend", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 3: Final detection result
        panel3 = frame.copy()
        
        if hand_info['detected'] and hand_info['contour'] is not None:
            # Draw selected hand
            contour_offset = hand_info['contour'].copy()
            contour_offset[:, :, 1] += roi_y
            cv2.drawContours(panel3, [contour_offset], -1, (0, 255, 0), 3)
            
            # Draw centroid
            cx, cy = hand_info['x'], hand_info['y']
            cv2.circle(panel3, (cx, cy), 15, (255, 0, 255), -1)
            cv2.circle(panel3, (cx, cy), 20, (255, 255, 255), 2)
            
            # Draw bounding box
            x, y, w, h = cv2.boundingRect(hand_info['contour'])
            cv2.rectangle(panel3, (x, y + roi_y), (x + w, y + h + roi_y), (255, 255, 0), 2)
            
            # Show area
            area = cv2.contourArea(hand_info['contour'])
            cv2.putText(panel3, f"Area: {int(area)}", (x, y + roi_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                       
            # Show finger count
            if 'finger_count' in hand_info:
                 cv2.putText(panel3, f"Fingers: {hand_info['finger_count']}", (x, y + roi_y - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(panel3, "NO VALID HAND DETECTED", (frame_width // 2 - 200, frame_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        cv2.line(panel3, (0, roi_y), (frame_width, roi_y), (0, 255, 255), 2)
        cv2.putText(panel3, "3. Final Hand Detection", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 4: Info
        panel4 = np.zeros_like(frame)
        
        info_y = 80
        cv2.putText(panel4, "STEP 2: Area Filtering", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        info_y += line_height + 20
        
        cv2.putText(panel4, f"FPS: {self.fps:.1f}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += line_height
        
        cv2.putText(panel4, f"Min Area: {self.detector.min_area}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += line_height
        
        cv2.putText(panel4, f"Max Area: {self.detector.max_area}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += line_height + 20
        
        cv2.putText(panel4, "This step filters contours", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        info_y += line_height - 5
        cv2.putText(panel4, "by area to remove:", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        info_y += line_height
        
        cv2.putText(panel4, "- Small noise", (40, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += line_height - 5
        cv2.putText(panel4, "- Large backgrounds", (40, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += line_height - 5
        cv2.putText(panel4, "- Face regions", (40, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controls
        info_y = frame_height - 100
        cv2.putText(panel4, "Controls:", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        info_y += line_height - 5
        cv2.putText(panel4, "Q - Quit", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(panel4, "4. Info & Purpose", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Combine panels
        top_row = np.hstack([panel1, panel2])
        bottom_row = np.hstack([panel3, panel4])
        combined = np.vstack([top_row, bottom_row])
        
        # Scale down for better screen fit (50% size)
        scale = 0.5
        new_width = int(combined.shape[1] * scale)
        new_height = int(combined.shape[0] * scale)
        combined = cv2.resize(combined, (new_width, new_height))
        
        return combined
    
    def run(self):
        """Run the test visualization."""
        print("=" * 80)
        print("STEP 2: CONTOUR AREA FILTERING TEST")
        print("=" * 80)
        print("This test demonstrates:")
        print("  - Classification of contours by area")
        print("  - Rejection of too small contours (noise)")
        print("  - Rejection of too large contours (background)")
        print("  - Selection of largest valid contour as hand")
        print()
        print("Watch how different sized regions are filtered!")
        print("=" * 80)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (960, 720))
            
            # Get contour information
            contour_info = self.get_all_contours_info(frame)
            
            # Detect hand
            hand_info = self.detector.detect_hand(frame, use_smoothing=False)
            
            # Count fingers if hand detected
            if hand_info['detected']:
                hand_info['finger_count'] = self.detector.count_fingers(hand_info['contour'])
            
            # Calculate FPS
            self.calculate_fps()
            
            # Create visualization
            viz = self.create_visualization(frame, contour_info, hand_info)
            
            # Show visualization
            cv2.imshow('Step 2: Contour Area Filtering', viz)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nTest completed!")

if __name__ == '__main__':
    test = TestStep2AreaFilter()
    test.run()
