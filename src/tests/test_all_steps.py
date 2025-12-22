import cv2
import numpy as np
from src.modules.hand_detector import HandDetector
from src.modules.column_mapper import ColumnMapper
from src.modules.renderer import Renderer

class TestAllSteps:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize hand detector with all improvements
        self.detector = HandDetector(
            roi_height_ratio=0.5,      # Step 1: ROI restriction
            min_area=5000,              # Step 2: Area filtering
            max_area=50000              # Step 2: Area filtering
        )
        self.detector.smoothing_alpha = 0.3  # Step 3: Temporal smoothing
        
        # Initialize column mapper (3 columns)
        self.mapper = ColumnMapper(num_cols=3, smooth=True)  # Step 5: 3 columns
        
        # Initialize renderer
        self.renderer = Renderer(width=960, height=720, num_cols=3)
        
        # Game state
        self.basket_col = 1  # Start at center
        
        # Detection method toggle
        self.use_palm_detection = True  # Step 4: Distance transform
        
        # FPS calculation
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = cv2.getTickCount()
        
        # Column labels
        self.column_labels = ["LEFT", "CENTER", "RIGHT"]
    
    def calculate_fps(self):
        self.frame_count += 1
        if self.frame_count >= 10:
            current_time = cv2.getTickCount()
            elapsed = (current_time - self.fps_start_time) / cv2.getTickFrequency()
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def create_info_panel(self, width, height, hand_info):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        y_pos = 60
        line_height = 35
        
        # Title
        cv2.putText(panel, "HAND DETECTION - ALL IMPROVEMENTS", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_pos += line_height + 20
        
        # FPS
        cv2.putText(panel, f"FPS: {self.fps:.1f}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_height + 20
        
        # Active features
        cv2.putText(panel, "ACTIVE FEATURES:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_pos += line_height + 10
        
        features = [
            ("1. ROI Restriction", f"Lower {int(self.detector.roi_height_ratio * 100)}%", (0, 255, 0)),
            ("2. Area Filtering", f"{self.detector.min_area}-{self.detector.max_area}px", (0, 255, 0)),
            ("3. Temporal Smoothing", f"α={self.detector.smoothing_alpha}", (0, 255, 0)),
            ("4. Palm Detection", "Distance Transform" if self.use_palm_detection else "Centroid", (0, 255, 0)),
            ("5. Column Layout", "3 Columns", (0, 255, 0))
        ]
        
        for feature, value, color in features:
            cv2.putText(panel, f"{feature}:", (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_pos += line_height - 10
            cv2.putText(panel, f"  {value}", (60, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_pos += line_height - 5
        
        y_pos += 20
        
        # Current status
        cv2.putText(panel, "CURRENT STATUS:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_pos += line_height + 10
        
        if hand_info['detected']:
            cv2.putText(panel, "Hand Detected: YES", (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += line_height
            
            cv2.putText(panel, f"Position: ({hand_info['x']}, {hand_info['y']})", (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += line_height - 5
            
            if 'palm_radius' in hand_info and hand_info['palm_radius'] > 0:
                cv2.putText(panel, f"Palm Radius: {hand_info['palm_radius']:.1f}px", (40, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_pos += line_height - 5
            
            cv2.putText(panel, f"Basket: {self.column_labels[self.basket_col]}", (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(panel, "Hand Detected: NO", (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Controls
        y_pos = height - 200
        cv2.putText(panel, "CONTROLS:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_pos += line_height
        
        controls = [
            "Q - Quit",
            "R - Reset",
            "P - Toggle Palm/Centroid",
            "+ - Increase smoothing",
            "- - Decrease smoothing"
        ]
        
        for control in controls:
            cv2.putText(panel, control, (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += line_height - 10
        
        return panel
    
    def create_visualization(self, frame, hand_info):
        frame_height, frame_width = frame.shape[:2]
        roi_y = hand_info['roi_start_y']
        
        # Main panel: Frame with all overlays
        main_panel = frame.copy()
        
        # Draw ROI boundary
        cv2.rectangle(main_panel, (0, roi_y), (frame_width, frame_height), (0, 255, 255), 3)
        cv2.putText(main_panel, "DETECTION ROI", (10, roi_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw column dividers
        col_width = frame_width / 3
        for i in range(1, 3):
            x = int(i * col_width)
            cv2.line(main_panel, (x, 0), (x, frame_height), (150, 150, 150), 2)
        
        # Highlight active column
        active_x1 = int(self.basket_col * col_width)
        active_x2 = int((self.basket_col + 1) * col_width)
        overlay = main_panel.copy()
        cv2.rectangle(overlay, (active_x1, 0), (active_x2, frame_height), (0, 255, 0), -1)
        main_panel = cv2.addWeighted(overlay, 0.15, main_panel, 0.85, 0)
        cv2.rectangle(main_panel, (active_x1, 0), (active_x2, frame_height), (0, 255, 0), 3)
        
        # Draw column labels
        for i in range(3):
            x_center = int(i * col_width + col_width / 2)
            label = self.column_labels[i]
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = x_center - text_size[0] // 2
            
            if i == self.basket_col:
                cv2.putText(main_panel, label, (text_x, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
            else:
                cv2.putText(main_panel, label, (text_x, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Draw hand detection
        if hand_info['detected']:
            # Draw contour
            if hand_info['contour'] is not None:
                contour_offset = hand_info['contour'].copy()
                contour_offset[:, :, 1] += roi_y
                cv2.drawContours(main_panel, [contour_offset], -1, (0, 255, 0), 3)
            
            # Draw hand center
            cx, cy = hand_info['x'], hand_info['y']
            
            # If using palm detection, draw palm circle
            if self.use_palm_detection and 'palm_radius' in hand_info:
                palm_radius = int(hand_info['palm_radius'])
                cv2.circle(main_panel, (cx, cy), palm_radius, (255, 0, 255), 2)
            
            # Draw center point
            cv2.circle(main_panel, (cx, cy), 18, (0, 255, 0), -1)
            cv2.circle(main_panel, (cx, cy), 24, (255, 255, 255), 3)
            
            # Draw crosshair
            cv2.line(main_panel, (cx - 35, cy), (cx + 35, cy), (255, 255, 255), 3)
            cv2.line(main_panel, (cx, cy - 35), (cx, cy + 35), (255, 255, 255), 3)
            
            # Draw vertical line to show x-position
            cv2.line(main_panel, (cx, 0), (cx, frame_height), (0, 255, 0), 2)
            
            # Position label
            cv2.putText(main_panel, f"({cx}, {cy})", (cx - 60, cy - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw basket (net)
        self.renderer.draw_net(main_panel, self.basket_col)
        
        # Info panel
        info_panel = self.create_info_panel(frame_width, frame_height, hand_info)
        
        # Combine panels side by side
        combined = np.hstack([main_panel, info_panel])
        
        return combined
    
    def run(self):
        """Run the comprehensive test."""
        print("=" * 80)
        print("COMPREHENSIVE HAND DETECTION TEST - ALL IMPROVEMENTS")
        print("=" * 80)
        print()
        print("This test demonstrates ALL implemented improvements:")
        print()
        print("  STEP 1: ROI Restriction")
        print("    - Detection limited to lower 50% of frame")
        print("    - Eliminates face detection")
        print("    - Reduces background interference")
        print()
        print("  STEP 2: Contour Area Filtering")
        print("    - Filters contours by size (5000-50000 pixels)")
        print("    - Removes small noise and large backgrounds")
        print("    - Only valid hand-sized regions detected")
        print()
        print("  STEP 3: Temporal Smoothing")
        print("    - Exponential moving average (α=0.3)")
        print("    - Reduces jitter and sudden jumps")
        print("    - Smooth basket movement")
        print()
        print("  STEP 4: Distance Transform Palm Detection")
        print("    - Finds palm center (most stable part)")
        print("    - More stable than centroid when fingers move")
        print("    - Toggle with 'P' key")
        print()
        print("  STEP 5: 3-Column Layout")
        print("    - Wider columns (320px vs 240px)")
        print("    - Easier to control")
        print("    - Less sensitivity to hand jitter")
        print()
        print("=" * 80)
        print("Move your hand in the lower half of the camera view!")
        print("=" * 80)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (960, 720))
            
            # Detect hand with chosen method
            if self.use_palm_detection:
                hand_info = self.detector.detect_hand_with_palm_center(frame, use_smoothing=True)
            else:
                hand_info = self.detector.detect_hand(frame, use_smoothing=True)
            
            # Map to column
            if hand_info['detected'] and hand_info['x'] is not None:
                self.basket_col = self.mapper.map_to_column(hand_info['x'], 960)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Create visualization
            viz = self.create_visualization(frame, hand_info)
            
            # Show visualization
            cv2.imshow('All Improvements - Hand Detection', viz)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.detector.reset()
                self.mapper.reset()
                self.basket_col = 1
                print("Reset")
            elif key == ord('p'):
                self.use_palm_detection = not self.use_palm_detection
                method = "Distance Transform" if self.use_palm_detection else "Centroid"
                print(f"Detection method: {method}")
            elif key == ord('+') or key == ord('='):
                self.detector.smoothing_alpha = min(1.0, self.detector.smoothing_alpha + 0.05)
                print(f"Smoothing α: {self.detector.smoothing_alpha:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.detector.smoothing_alpha = max(0.05, self.detector.smoothing_alpha - 0.05)
                print(f"Smoothing α: {self.detector.smoothing_alpha:.2f}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nTest completed!")
        print("All improvements are working together for robust hand detection!")

if __name__ == '__main__':
    test = TestAllSteps()
    test.run()
