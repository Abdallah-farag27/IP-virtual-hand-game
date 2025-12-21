import cv2
import numpy as np
from hand_detector import HandDetector
from column_mapper import ColumnMapper
from renderer import Renderer

class TestStep5ThreeColumns:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize hand detector with all improvements
        self.detector = HandDetector(
            roi_height_ratio=0.5,
            min_area=5000,
            max_area=50000
        )
        self.detector.smoothing_alpha = 0.3
        
        # Initialize mappers for both 3 and 4 columns
        self.mapper_3col = ColumnMapper(num_cols=3, smooth=True)
        self.mapper_4col = ColumnMapper(num_cols=4, smooth=True)
        
        # Initialize renderers
        self.renderer_3col = Renderer(width=960, height=720, num_cols=3)
        self.renderer_4col = Renderer(width=960, height=720, num_cols=4)
        
        # Current basket positions
        self.basket_3col = 1  # Start at center
        self.basket_4col = 1
        
        # FPS calculation
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = cv2.getTickCount()
        
        # Column labels
        self.labels_3col = ["LEFT", "CENTER", "RIGHT"]
        self.labels_4col = ["LEFT", "CENTER-LEFT", "CENTER-RIGHT", "RIGHT"]
    
    def calculate_fps(self):
        self.frame_count += 1
        if self.frame_count >= 10:
            current_time = cv2.getTickCount()
            elapsed = (current_time - self.fps_start_time) / cv2.getTickFrequency()
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def draw_column_overlay(self, frame, num_cols, basket_col, labels):
        frame_height, frame_width = frame.shape[:2]
        col_width = frame_width / num_cols
        
        overlay = frame.copy()
        
        # Draw column dividers
        for i in range(1, num_cols):
            x = int(i * col_width)
            cv2.line(overlay, (x, 0), (x, frame_height), (100, 100, 100), 2)
        
        # Highlight active column
        active_x1 = int(basket_col * col_width)
        active_x2 = int((basket_col + 1) * col_width)
        cv2.rectangle(overlay, (active_x1, 0), (active_x2, frame_height), 
                     (0, 255, 0), 3)
        
        # Draw column labels
        for i in range(num_cols):
            x_center = int(i * col_width + col_width / 2)
            label = labels[i]
            
            # Background for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = x_center - text_size[0] // 2
            text_y = 40
            
            # Highlight active column label
            if i == basket_col:
                cv2.rectangle(overlay, (text_x - 10, text_y - 30), 
                            (text_x + text_size[0] + 10, text_y + 10),
                            (0, 255, 0), -1)
                cv2.putText(overlay, label, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            else:
                cv2.putText(overlay, label, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Blend overlay
        result = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        return result
    
    def create_visualization(self, frame, hand_info):
        frame_height, frame_width = frame.shape[:2]
        roi_y = hand_info['roi_start_y']
        
        # Panel 1: 3-column layout
        panel1 = frame.copy()
        
        # Draw ROI boundary
        cv2.line(panel1, (0, roi_y), (frame_width, roi_y), (0, 255, 255), 2)
        
        # Draw hand detection
        if hand_info['detected']:
            cx, cy = hand_info['x'], hand_info['y']
            
            if hand_info['contour'] is not None:
                contour_offset = hand_info['contour'].copy()
                contour_offset[:, :, 1] += roi_y
                cv2.drawContours(panel1, [contour_offset], -1, (0, 255, 0), 2)
            
            cv2.circle(panel1, (cx, cy), 15, (0, 255, 0), -1)
            cv2.circle(panel1, (cx, cy), 20, (255, 255, 255), 2)
            
            # Draw vertical line showing x-position
            cv2.line(panel1, (cx, 0), (cx, frame_height), (0, 255, 0), 2)
        
        # Draw column overlay
        panel1 = self.draw_column_overlay(panel1, 3, self.basket_3col, self.labels_3col)
        
        cv2.putText(panel1, "1. 3-Column Layout (NEW)", (10, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 2: 4-column layout (for comparison)
        panel2 = frame.copy()
        
        # Draw ROI boundary
        cv2.line(panel2, (0, roi_y), (frame_width, roi_y), (0, 255, 255), 2)
        
        # Draw hand detection
        if hand_info['detected']:
            cx, cy = hand_info['x'], hand_info['y']
            
            if hand_info['contour'] is not None:
                contour_offset = hand_info['contour'].copy()
                contour_offset[:, :, 1] += roi_y
                cv2.drawContours(panel2, [contour_offset], -1, (255, 165, 0), 2)
            
            cv2.circle(panel2, (cx, cy), 15, (255, 165, 0), -1)
            cv2.circle(panel2, (cx, cy), 20, (255, 255, 255), 2)
            
            # Draw vertical line
            cv2.line(panel2, (cx, 0), (cx, frame_height), (255, 165, 0), 2)
        
        # Draw column overlay
        panel2 = self.draw_column_overlay(panel2, 4, self.basket_4col, self.labels_4col)
        
        cv2.putText(panel2, "2. 4-Column Layout (OLD)", (10, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 3: Column width comparison
        panel3 = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Draw 3-column bars
        bar_height = 80
        bar_y = 100
        col_width_3 = frame_width / 3
        
        cv2.putText(panel3, "3 Columns (Wider):", (20, bar_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        for i in range(3):
            x1 = int(i * col_width_3) + 10
            x2 = int((i + 1) * col_width_3) - 10
            color = (0, 255, 0) if i == self.basket_3col else (100, 100, 100)
            cv2.rectangle(panel3, (x1, bar_y), (x2, bar_y + bar_height), color, -1)
            cv2.rectangle(panel3, (x1, bar_y), (x2, bar_y + bar_height), (255, 255, 255), 2)
            
            # Label
            label = self.labels_3col[i]
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = (x1 + x2) // 2 - text_size[0] // 2
            cv2.putText(panel3, label, (text_x, bar_y + bar_height // 2 + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Width label
            width_px = int(col_width_3)
            cv2.putText(panel3, f"{width_px}px", (text_x, bar_y + bar_height + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw 4-column bars
        bar_y = 300
        col_width_4 = frame_width / 4
        
        cv2.putText(panel3, "4 Columns (Narrower):", (20, bar_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        
        for i in range(4):
            x1 = int(i * col_width_4) + 10
            x2 = int((i + 1) * col_width_4) - 10
            color = (255, 165, 0) if i == self.basket_4col else (100, 100, 100)
            cv2.rectangle(panel3, (x1, bar_y), (x2, bar_y + bar_height), color, -1)
            cv2.rectangle(panel3, (x1, bar_y), (x2, bar_y + bar_height), (255, 255, 255), 2)
            
            # Label
            label = self.labels_4col[i]
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = (x1 + x2) // 2 - text_size[0] // 2
            cv2.putText(panel3, label, (text_x, bar_y + bar_height // 2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Width label
            width_px = int(col_width_4)
            cv2.putText(panel3, f"{width_px}px", (text_x, bar_y + bar_height + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Advantages
        adv_y = 500
        cv2.putText(panel3, "ADVANTAGES OF 3 COLUMNS:", (20, adv_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        adv_y += 45
        
        advantages = [
            "- Wider columns = less sensitivity",
            "- Easier to control basket",
            "- Reduced jitter effect",
            "- Simpler gameplay",
            "- More forgiving for players"
        ]
        
        for adv in advantages:
            cv2.putText(panel3, adv, (40, adv_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            adv_y += 35
        
        cv2.putText(panel3, "3. Column Width Comparison", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 4: Info
        panel4 = np.zeros_like(frame)
        
        y_pos = 80
        line_height = 35
        
        cv2.putText(panel4, "STEP 5: 3-Column Layout", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        y_pos += line_height + 20
        
        cv2.putText(panel4, f"FPS: {self.fps:.1f}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_height + 20
        
        # Current positions
        cv2.putText(panel4, "CURRENT POSITIONS:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_pos += line_height + 10
        
        if hand_info['detected']:
            cv2.putText(panel4, f"Hand X: {hand_info['x']}", (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += line_height
        
        cv2.putText(panel4, f"3-Col Basket: {self.labels_3col[self.basket_3col]}", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += line_height
        
        cv2.putText(panel4, f"4-Col Basket: {self.labels_4col[self.basket_4col]}", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        y_pos += line_height + 20
        
        # Column widths
        cv2.putText(panel4, "COLUMN WIDTHS:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_pos += line_height + 10
        
        cv2.putText(panel4, f"3 Columns: {int(frame_width/3)}px each", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += line_height
        
        cv2.putText(panel4, f"4 Columns: {int(frame_width/4)}px each", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        y_pos += line_height
        
        diff = int(frame_width/3) - int(frame_width/4)
        cv2.putText(panel4, f"Difference: +{diff}px (+{diff*100//int(frame_width/4)}%)", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Controls
        y_pos = frame_height - 100
        cv2.putText(panel4, "Controls:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += line_height - 5
        
        cv2.putText(panel4, "Q - Quit", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_height - 10
        
        cv2.putText(panel4, "R - Reset", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(panel4, "4. Info & Statistics", (10, 30),
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
        print("STEP 5: 3-COLUMN LAYOUT TEST")
        print("=" * 80)
        print("This test demonstrates:")
        print("  - 3-column layout (Left - Center - Right)")
        print("  - Comparison with 4-column layout")
        print("  - Wider columns for easier control")
        print("  - Reduced sensitivity to hand jitter")
        print()
        print("Move your hand left and right to see the difference!")
        print("=" * 80)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (960, 720))
            
            # Detect hand with all improvements
            hand_info = self.detector.detect_hand_with_palm_center(frame, use_smoothing=True)
            
            # Map to columns
            if hand_info['detected'] and hand_info['x'] is not None:
                self.basket_3col = self.mapper_3col.map_to_column(hand_info['x'], 960)
                self.basket_4col = self.mapper_4col.map_to_column(hand_info['x'], 960)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Create visualization
            viz = self.create_visualization(frame, hand_info)
            
            # Show visualization
            cv2.imshow('Step 5: 3-Column Layout', viz)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.detector.reset()
                self.mapper_3col.reset()
                self.mapper_4col.reset()
                print("Reset")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nTest completed!")

if __name__ == '__main__':
    test = TestStep5ThreeColumns()
    test.run()
