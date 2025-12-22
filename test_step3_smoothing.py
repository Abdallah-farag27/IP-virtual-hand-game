import cv2
import numpy as np
from src.modules.hand_detector import HandDetector
import collections

class TestStep3Smoothing:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize two detectors for comparison
        self.detector_raw = HandDetector(
            roi_height_ratio=0.5,
            min_area=5000,
            max_area=50000
        )
        self.detector_raw.smoothing_alpha = 1.0  # No smoothing
        
        self.detector_smooth = HandDetector(
            roi_height_ratio=0.5,
            min_area=5000,
            max_area=50000
        )
        self.detector_smooth.smoothing_alpha = 0.3  # Smoothing enabled
        
        # History tracking (last 100 frames)
        self.history_length = 100
        self.raw_x_history = collections.deque(maxlen=self.history_length)
        self.smooth_x_history = collections.deque(maxlen=self.history_length)
        self.frame_numbers = collections.deque(maxlen=self.history_length)
        self.current_frame = 0
        
        # Jitter calculation
        self.raw_jitter = 0
        self.smooth_jitter = 0
        
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
    
    def calculate_jitter(self, history):
        if len(history) < 2:
            return 0
        
        diffs = [abs(history[i] - history[i-1]) for i in range(1, len(history))]
        return sum(diffs) / len(diffs) if diffs else 0
    
    def draw_graph(self, width, height):
        graph = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(self.raw_x_history) < 2:
            cv2.putText(graph, "Collecting data...", (width // 2 - 100, height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return graph
        
        # Draw grid
        grid_color = (50, 50, 50)
        for i in range(0, height, height // 10):
            cv2.line(graph, (0, i), (width, i), grid_color, 1)
        for i in range(0, width, width // 10):
            cv2.line(graph, (i, 0), (i, height), grid_color, 1)
        
        # Draw axes labels
        cv2.putText(graph, "Frame", (width - 60, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(graph, "X Position", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Normalize x-coordinates to graph height
        min_x = 0
        max_x = 960
        
        def normalize_y(x):
            """Convert x-coordinate to graph y-coordinate."""
            normalized = (x - min_x) / (max_x - min_x)
            return int(height - (normalized * (height - 40)) - 20)
        
        # Draw raw position line (red)
        raw_points = [(int(i * width / self.history_length), normalize_y(x)) 
                      for i, x in enumerate(self.raw_x_history)]
        
        for i in range(1, len(raw_points)):
            cv2.line(graph, raw_points[i-1], raw_points[i], (0, 0, 255), 2)
        
        # Draw smoothed position line (green)
        smooth_points = [(int(i * width / self.history_length), normalize_y(x)) 
                        for i, x in enumerate(self.smooth_x_history)]
        
        for i in range(1, len(smooth_points)):
            cv2.line(graph, smooth_points[i-1], smooth_points[i], (0, 255, 0), 2)
        
        # Legend
        cv2.line(graph, (10, height - 60), (40, height - 60), (0, 0, 255), 2)
        cv2.putText(graph, "Raw", (45, height - 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.line(graph, (10, height - 40), (40, height - 40), (0, 255, 0), 2)
        cv2.putText(graph, "Smoothed", (45, height - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return graph
    
    def create_visualization(self, frame, raw_info, smooth_info):
        frame_height, frame_width = frame.shape[:2]
        roi_y = raw_info['roi_start_y']
        
        # Panel 1: Raw detection
        panel1 = frame.copy()
        cv2.line(panel1, (0, roi_y), (frame_width, roi_y), (0, 255, 255), 2)
        
        if raw_info['detected']:
            cx, cy = raw_info['x'], raw_info['y']
            
            # Draw contour
            if raw_info['contour'] is not None:
                contour_offset = raw_info['contour'].copy()
                contour_offset[:, :, 1] += roi_y
                cv2.drawContours(panel1, [contour_offset], -1, (0, 0, 255), 2)
            
            # Draw centroid
            cv2.circle(panel1, (cx, cy), 15, (0, 0, 255), -1)
            cv2.circle(panel1, (cx, cy), 20, (255, 255, 255), 2)
            
            # Draw vertical line to show x-position
            cv2.line(panel1, (cx, 0), (cx, frame_height), (0, 0, 255), 2)
            cv2.putText(panel1, f"X: {cx}", (cx - 30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.putText(panel1, "1. Raw Detection (No Smoothing)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 2: Smoothed detection
        panel2 = frame.copy()
        cv2.line(panel2, (0, roi_y), (frame_width, roi_y), (0, 255, 255), 2)
        
        if smooth_info['detected']:
            cx, cy = smooth_info['x'], smooth_info['y']
            
            # Draw contour
            if smooth_info['contour'] is not None:
                contour_offset = smooth_info['contour'].copy()
                contour_offset[:, :, 1] += roi_y
                cv2.drawContours(panel2, [contour_offset], -1, (0, 255, 0), 2)
            
            # Draw centroid
            cv2.circle(panel2, (cx, cy), 15, (0, 255, 0), -1)
            cv2.circle(panel2, (cx, cy), 20, (255, 255, 255), 2)
            
            # Draw vertical line to show x-position
            cv2.line(panel2, (cx, 0), (cx, frame_height), (0, 255, 0), 2)
            cv2.putText(panel2, f"X: {cx}", (cx - 30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(panel2, "2. Smoothed Detection (α=0.3)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 3: Position history graph
        panel3 = self.draw_graph(frame_width, frame_height)
        cv2.putText(panel3, "3. Position History", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 4: Statistics and info
        panel4 = np.zeros_like(frame)
        
        y_pos = 80
        line_height = 35
        
        cv2.putText(panel4, "STEP 3: Temporal Smoothing", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        y_pos += line_height + 20
        
        cv2.putText(panel4, f"FPS: {self.fps:.1f}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_height
        
        cv2.putText(panel4, f"Frame: {self.current_frame}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_height + 20
        
        # Jitter comparison
        cv2.putText(panel4, "JITTER ANALYSIS:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_pos += line_height + 10
        
        cv2.putText(panel4, f"Raw Jitter: {self.raw_jitter:.2f} px", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += line_height
        
        cv2.putText(panel4, f"Smoothed Jitter: {self.smooth_jitter:.2f} px", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += line_height
        
        if self.raw_jitter > 0:
            reduction = ((self.raw_jitter - self.smooth_jitter) / self.raw_jitter) * 100
            cv2.putText(panel4, f"Reduction: {reduction:.1f}%", (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y_pos += line_height + 20
        
        # Smoothing explanation
        cv2.putText(panel4, "Exponential Smoothing:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y_pos += line_height - 5
        
        cv2.putText(panel4, "smoothed = α × raw +", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += line_height - 10
        
        cv2.putText(panel4, "           (1-α) × prev", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += line_height
        
        cv2.putText(panel4, f"α = {self.detector_smooth.smoothing_alpha}", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += line_height + 10
        
        cv2.putText(panel4, "Lower α = More smoothing", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls
        y_pos = frame_height - 140
        cv2.putText(panel4, "Controls:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += line_height - 5
        
        cv2.putText(panel4, "Q - Quit", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_height - 10
        
        cv2.putText(panel4, "+ - Increase α (less smooth)", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_height - 10
        
        cv2.putText(panel4, "- - Decrease α (more smooth)", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_height - 10
        
        cv2.putText(panel4, "R - Reset", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(panel4, "4. Statistics & Info", (10, 30),
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
        print("STEP 3: TEMPORAL SMOOTHING TEST")
        print("=" * 80)
        print("This test demonstrates:")
        print("  - Exponential moving average smoothing")
        print("  - Jitter reduction comparison")
        print("  - Position history visualization")
        print("  - Adjustable smoothing factor (α)")
        print()
        print("Move your hand quickly to see the smoothing effect!")
        print("=" * 80)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (960, 720))
            
            # Detect with both detectors
            raw_info = self.detector_raw.detect_hand(frame, use_smoothing=False)
            smooth_info = self.detector_smooth.detect_hand(frame, use_smoothing=True)
            
            # Update history
            if raw_info['detected'] and raw_info['x'] is not None:
                self.raw_x_history.append(raw_info['x'])
            
            if smooth_info['detected'] and smooth_info['x'] is not None:
                self.smooth_x_history.append(smooth_info['x'])
            
            self.frame_numbers.append(self.current_frame)
            self.current_frame += 1
            
            # Calculate jitter
            self.raw_jitter = self.calculate_jitter(self.raw_x_history)
            self.smooth_jitter = self.calculate_jitter(self.smooth_x_history)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Create visualization
            viz = self.create_visualization(frame, raw_info, smooth_info)
            
            # Show visualization
            cv2.imshow('Step 3: Temporal Smoothing', viz)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                self.detector_smooth.smoothing_alpha = min(1.0, self.detector_smooth.smoothing_alpha + 0.05)
                print(f"Smoothing α: {self.detector_smooth.smoothing_alpha:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.detector_smooth.smoothing_alpha = max(0.05, self.detector_smooth.smoothing_alpha - 0.05)
                print(f"Smoothing α: {self.detector_smooth.smoothing_alpha:.2f}")
            elif key == ord('r'):
                self.detector_raw.reset()
                self.detector_smooth.reset()
                self.raw_x_history.clear()
                self.smooth_x_history.clear()
                self.frame_numbers.clear()
                self.current_frame = 0
                print("Reset")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nTest completed!")
        print(f"Final jitter reduction: {((self.raw_jitter - self.smooth_jitter) / self.raw_jitter * 100):.1f}%")

if __name__ == '__main__':
    test = TestStep3Smoothing()
    test.run()
