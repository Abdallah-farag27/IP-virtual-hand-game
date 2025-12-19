import cv2
import numpy as np
from hand_detector import HandDetector

class TestStep4DistanceTransform:
    """
    Test Step 4: Distance Transform Palm Detection
    
    This test shows:
    1. Centroid-based detection (traditional method)
    2. Distance transform-based palm detection (more stable)
    3. Distance transform heatmap visualization
    4. Comparison of stability between methods
    """
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize two detectors for comparison
        self.detector_centroid = HandDetector(
            roi_height_ratio=0.5,
            min_area=5000,
            max_area=50000
        )
        
        self.detector_palm = HandDetector(
            roi_height_ratio=0.5,
            min_area=5000,
            max_area=50000
        )
        
        # FPS calculation
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = cv2.getTickCount()
    
    def calculate_fps(self):
        """Calculate and update FPS."""
        self.frame_count += 1
        if self.frame_count >= 10:
            current_time = cv2.getTickCount()
            elapsed = (current_time - self.fps_start_time) / cv2.getTickFrequency()
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def create_heatmap(self, dist_transform, width, height):
        """
        Create colored heatmap from distance transform.
        
        Args:
            dist_transform: Distance transform image
            width: Output width
            height: Output height
            
        Returns:
            Colored heatmap image
        """
        if dist_transform is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        # Normalize to 0-255
        normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        
        # Apply color map (COLORMAP_JET for heat visualization)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        return heatmap
    
    def create_visualization(self, frame, centroid_info, palm_info):
        """
        Create multi-panel visualization.
        
        Args:
            frame: Original frame
            centroid_info: Centroid-based detection result
            palm_info: Palm center detection result
            
        Returns:
            Combined visualization image
        """
        frame_height, frame_width = frame.shape[:2]
        roi_y = centroid_info['roi_start_y']
        
        # Panel 1: Centroid-based detection
        panel1 = frame.copy()
        cv2.line(panel1, (0, roi_y), (frame_width, roi_y), (0, 255, 255), 2)
        
        if centroid_info['detected']:
            # Draw contour
            if centroid_info['contour'] is not None:
                contour_offset = centroid_info['contour'].copy()
                contour_offset[:, :, 1] += roi_y
                cv2.drawContours(panel1, [contour_offset], -1, (0, 165, 255), 2)
            
            # Draw centroid
            cx, cy = centroid_info['x'], centroid_info['y']
            cv2.circle(panel1, (cx, cy), 12, (0, 165, 255), -1)
            cv2.circle(panel1, (cx, cy), 18, (255, 255, 255), 2)
            
            # Draw crosshair
            cv2.line(panel1, (cx - 25, cy), (cx + 25, cy), (255, 255, 255), 2)
            cv2.line(panel1, (cx, cy - 25), (cx, cy + 25), (255, 255, 255), 2)
            
            cv2.putText(panel1, "Centroid", (cx - 40, cy - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        cv2.putText(panel1, "1. Centroid Method", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 2: Distance transform palm detection
        panel2 = frame.copy()
        cv2.line(panel2, (0, roi_y), (frame_width, roi_y), (0, 255, 255), 2)
        
        if palm_info['detected']:
            # Draw contour
            if palm_info['contour'] is not None:
                contour_offset = palm_info['contour'].copy()
                contour_offset[:, :, 1] += roi_y
                cv2.drawContours(panel2, [contour_offset], -1, (0, 255, 0), 2)
            
            # Draw palm center
            px, py = palm_info['x'], palm_info['y']
            palm_radius = int(palm_info['palm_radius'])
            
            # Draw palm circle (shows detected palm region)
            cv2.circle(panel2, (px, py), palm_radius, (255, 0, 255), 2)
            cv2.circle(panel2, (px, py), 12, (0, 255, 0), -1)
            cv2.circle(panel2, (px, py), 18, (255, 255, 255), 2)
            
            # Draw crosshair
            cv2.line(panel2, (px - 25, py), (px + 25, py), (255, 255, 255), 2)
            cv2.line(panel2, (px, py - 25), (px, py + 25), (255, 255, 255), 2)
            
            cv2.putText(panel2, "Palm Center", (px - 50, py - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(panel2, f"R: {palm_radius}", (px - 30, py + palm_radius + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        cv2.putText(panel2, "2. Distance Transform Method", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 3: Distance transform heatmap
        panel3 = np.zeros_like(frame)
        
        if palm_info['distance_transform'] is not None:
            # Create heatmap
            heatmap = self.create_heatmap(palm_info['distance_transform'], 
                                         frame_width, 
                                         palm_info['distance_transform'].shape[0])
            
            # Place heatmap in ROI
            panel3[roi_y:roi_y + heatmap.shape[0], :] = heatmap
            
            # Mark palm center on heatmap
            if palm_info['detected']:
                px, py = palm_info['x'], palm_info['y']
                cv2.circle(panel3, (px, py), 15, (255, 255, 255), 3)
                cv2.circle(panel3, (px, py), 10, (0, 0, 0), -1)
                
                # Draw max distance value
                palm_radius = palm_info['palm_radius']
                cv2.putText(panel3, f"Max: {palm_radius:.1f}", (px + 25, py),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.line(panel3, (0, roi_y), (frame_width, roi_y), (255, 255, 255), 2)
        cv2.putText(panel3, "3. Distance Transform Heatmap", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add color scale legend
        scale_height = 200
        scale_width = 30
        scale_x = frame_width - scale_width - 20
        scale_y = roi_y + 50
        
        # Create gradient scale
        gradient = np.linspace(255, 0, scale_height).reshape(scale_height, 1)
        gradient = np.tile(gradient, (1, scale_width)).astype(np.uint8)
        gradient_colored = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        
        panel3[scale_y:scale_y + scale_height, scale_x:scale_x + scale_width] = gradient_colored
        cv2.rectangle(panel3, (scale_x, scale_y), (scale_x + scale_width, scale_y + scale_height),
                     (255, 255, 255), 2)
        
        cv2.putText(panel3, "Far", (scale_x - 40, scale_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel3, "Near", (scale_x - 45, scale_y + scale_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Panel 4: Info and comparison
        panel4 = np.zeros_like(frame)
        
        y_pos = 80
        line_height = 35
        
        cv2.putText(panel4, "STEP 4: Distance Transform", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        y_pos += line_height + 20
        
        cv2.putText(panel4, f"FPS: {self.fps:.1f}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_height + 20
        
        # Method comparison
        cv2.putText(panel4, "METHOD COMPARISON:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_pos += line_height + 10
        
        # Centroid method
        cv2.putText(panel4, "Centroid Method:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        y_pos += line_height - 5
        
        if centroid_info['detected']:
            cv2.putText(panel4, f"  Position: ({centroid_info['x']}, {centroid_info['y']})", (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(panel4, "  Not detected", (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_pos += line_height
        
        # Palm center method
        cv2.putText(panel4, "Palm Center Method:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += line_height - 5
        
        if palm_info['detected']:
            cv2.putText(panel4, f"  Position: ({palm_info['x']}, {palm_info['y']})", (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += line_height - 10
            cv2.putText(panel4, f"  Palm Radius: {palm_info['palm_radius']:.1f} px", (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(panel4, "  Not detected", (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_pos += line_height + 20
        
        # Advantages
        cv2.putText(panel4, "ADVANTAGES:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y_pos += line_height - 5
        
        cv2.putText(panel4, "- More stable when fingers", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += line_height - 10
        cv2.putText(panel4, "  are moving", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += line_height - 5
        
        cv2.putText(panel4, "- Focuses on palm center", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += line_height - 10
        cv2.putText(panel4, "  (most stable part)", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += line_height - 5
        
        cv2.putText(panel4, "- Less affected by hand", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += line_height - 10
        cv2.putText(panel4, "  shape changes", (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
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
        
        cv2.putText(panel4, "4. Info & Comparison", (10, 30),
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
        print("STEP 4: DISTANCE TRANSFORM PALM DETECTION TEST")
        print("=" * 80)
        print("This test demonstrates:")
        print("  - Distance transform for finding palm center")
        print("  - Comparison with centroid-based detection")
        print("  - Heatmap visualization of distance transform")
        print("  - Improved stability when fingers move")
        print()
        print("Try moving your fingers while keeping palm still!")
        print("Notice how palm center stays more stable than centroid.")
        print("=" * 80)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (960, 720))
            
            # Detect with both methods
            centroid_info = self.detector_centroid.detect_hand(frame, use_smoothing=False)
            palm_info = self.detector_palm.detect_hand_with_palm_center(frame, use_smoothing=False)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Create visualization
            viz = self.create_visualization(frame, centroid_info, palm_info)
            
            # Show visualization
            cv2.imshow('Step 4: Distance Transform Palm Detection', viz)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.detector_centroid.reset()
                self.detector_palm.reset()
                print("Reset")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nTest completed!")

if __name__ == '__main__':
    test = TestStep4DistanceTransform()
    test.run()
