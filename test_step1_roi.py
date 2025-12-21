import cv2
import numpy as np
from hand_detector import HandDetector

class TestStep1ROI:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize hand detector
        self.detector = HandDetector(
            roi_height_ratio=0.5,  # Lower 50% of frame
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
    
    def create_visualization(self, frame, hand_info):
        frame_height, frame_width = frame.shape[:2]
        
        # Panel 1: Original frame with ROI boundary
        panel1 = frame.copy()
        roi_y = hand_info['roi_start_y']
        cv2.line(panel1, (0, roi_y), (frame_width, roi_y), (0, 255, 255), 3)
        cv2.putText(panel1, "ROI Boundary", (10, roi_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(panel1, "1. Original + ROI", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 2: Skin mask
        skin_mask_colored = cv2.cvtColor(hand_info['skin_mask'], cv2.COLOR_GRAY2BGR)
        panel2 = np.zeros_like(frame)
        panel2[roi_y:, :] = skin_mask_colored
        cv2.putText(panel2, "2. Skin Detection Mask", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 3: Detected hand contour
        panel3 = frame.copy()
        
        if hand_info['detected'] and hand_info['contour'] is not None:
            # Draw contour (offset by ROI start)
            contour_offset = hand_info['contour'].copy()
            contour_offset[:, :, 1] += roi_y
            cv2.drawContours(panel3, [contour_offset], -1, (0, 255, 0), 3)
            
            # Draw centroid
            cx, cy = hand_info['x'], hand_info['y']
            cv2.circle(panel3, (cx, cy), 15, (255, 0, 255), -1)
            cv2.circle(panel3, (cx, cy), 20, (255, 255, 255), 2)
            
            # Draw crosshair
            cv2.line(panel3, (cx - 30, cy), (cx + 30, cy), (255, 255, 255), 2)
            cv2.line(panel3, (cx, cy - 30), (cx, cy + 30), (255, 255, 255), 2)
            
            # Show coordinates
            coord_text = f"Hand: ({cx}, {cy})"
            cv2.putText(panel3, coord_text, (cx - 80, cy - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show if smoothing is applied
            if 'raw_x' in hand_info and hand_info['raw_x'] != cx:
                cv2.putText(panel3, f"Raw X: {hand_info['raw_x']}, Smoothed X: {cx}",
                           (10, frame_height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(panel3, "NO HAND DETECTED", (frame_width // 2 - 150, frame_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        cv2.putText(panel3, "3. Hand Detection Result", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 4: Info panel
        panel4 = np.zeros_like(frame)
        
        # Draw info text
        info_y = 80
        line_height = 40
        
        cv2.putText(panel4, "STEP 1: ROI + Skin Detection", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        info_y += line_height + 20
        
        cv2.putText(panel4, f"FPS: {self.fps:.1f}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += line_height
        
        cv2.putText(panel4, f"ROI: Lower {int(self.detector.roi_height_ratio * 100)}%", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += line_height
        
        cv2.putText(panel4, f"Detected: {'YES' if hand_info['detected'] else 'NO'}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if hand_info['detected'] else (0, 0, 255), 2)
        info_y += line_height
        
        if hand_info['detected']:
            cv2.putText(panel4, f"Position: ({hand_info['x']}, {hand_info['y']})", (20, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += line_height
            
            if hand_info['contour'] is not None:
                area = cv2.contourArea(hand_info['contour'])
                cv2.putText(panel4, f"Contour Area: {int(area)}", (20, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                info_y += line_height
        
        # Instructions
        info_y = frame_height - 180
        cv2.putText(panel4, "Controls:", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        info_y += line_height
        
        cv2.putText(panel4, "Q - Quit", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += line_height - 5
        
        cv2.putText(panel4, "S - Toggle Smoothing", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += line_height - 5
        
        cv2.putText(panel4, "R - Reset", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(panel4, "4. Info & Controls", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Combine panels in 2x2 grid
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
        print("STEP 1: ROI RESTRICTION + SKIN DETECTION TEST")
        print("=" * 80)
        print("This test demonstrates:")
        print("  - ROI restriction to lower 50% of frame (eliminates face detection)")
        print("  - HSV-based skin color segmentation")
        print("  - Morphological operations for noise reduction")
        print("  - Contour area filtering")
        print("  - Temporal smoothing of hand position")
        print()
        print("Move your hand in the LOWER half of the camera view.")
        print("Your face (in upper half) should be ignored.")
        print("=" * 80)
        
        use_smoothing = True
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (960, 720))
            
            # Detect hand
            hand_info = self.detector.detect_hand(frame, use_smoothing=use_smoothing)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Create visualization
            viz = self.create_visualization(frame, hand_info)
            
            # Show visualization
            cv2.imshow('Step 1: ROI + Skin Detection', viz)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                use_smoothing = not use_smoothing
                print(f"Temporal smoothing: {'ON' if use_smoothing else 'OFF'}")
            elif key == ord('r'):
                self.detector.reset()
                print("Detector reset")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nTest completed!")

if __name__ == '__main__':
    test = TestStep1ROI()
    test.run()
