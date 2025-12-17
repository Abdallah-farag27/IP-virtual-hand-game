import cv2
import numpy as np
import time
from background_subtractor import BackgroundDetector

from preprocess import preprocess_frame
from skin_detector import detect_skin
from hand_contour import find_hand_contour

def create_synthetic_test_image():
    """Create a synthetic image with a skin-colored rectangle for testing."""
    img = np.zeros((720, 960, 3), dtype=np.uint8)
    # Background: Dark Gray
    img[:] = (50, 50, 50)
    # Skin color rectangle (Approximate skin color in BGR)
    # B: ~120, G: ~150, R: ~200
    cv2.rectangle(img, (300, 200), (660, 520), (120, 150, 200), -1)
    # Add some noise
    noise = np.random.randint(0, 50, (720, 960, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    return img

def main():
    print("Testing Hassan's + Wafa's Modules + Background Subtraction...")
    
    # Try opening webcam
    cap = cv2.VideoCapture(0)
    use_camera = cap.isOpened()
    
    # Initialize Background Detector
    bg_detector = BackgroundDetector()
    
    if use_camera:
        print("Camera found! Running live test.")
        print("Press 'q' to quit.")
        start_time = time.time()
    else:
        print("Camera not found. Using synthetic test image.")
        frame = create_synthetic_test_image()
    
    try:
        while True:
            if use_camera:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame.")
                    break
            
            # --- 1. Preprocessing ---
            t1 = time.time()
            
            # Use the same resize/flip logic as preprocess.py to ensure alignment
            # (Or modify preprocess to return it, but let's keep it simple here)
            width, height = 960, 720
            frame_resized = cv2.resize(frame, (width, height))
            frame_flipped = frame_resized[:, ::-1]
            
            # Now get HSV from the preprocessor (it does the blur etc internally)
            # ideally we should pass frame_flipped to it if it supported it, 
            # but preprocess_frame takes raw frame. 
            # Let's just use the output of preprocess_frame (HSV) for skin
            # AND use frame_flipped for background subtraction.
            hsv_frame = preprocess_frame(frame)
            
            # --- 2. Skin Detection ---
            skin_mask = detect_skin(hsv_frame)
            t2 = time.time()
            
            # --- 2.5 Background Subtraction ---
            # Get mask of moving objects using the ALIGNED frame (frame_flipped)
            moving_mask = bg_detector.apply(frame_flipped)
            
            # COMBINE: Pixel must be Skin (Color) AND Moving (Foreground)
            clean_mask = cv2.bitwise_and(skin_mask, skin_mask, mask=moving_mask)
            
            # --- 3. Contour Detection (Wafa's Task) ---
            # We pass the CLEANED mask to find the contour
            contour, bbox, hull = find_hand_contour(clean_mask)
            t3 = time.time()
            
            preprocess_time = (t2 - t1) * 1000
            contour_time = (t3 - t2) * 1000
            
            # --- Visualization ---
            display_h = 360
            display_w = 480
            
            # 1. Original (Small)
            orig_small = cv2.resize(frame, (display_w, display_h))
            
            # 2. Moving Mask (Visualizing what is "Moving")
            moving_small = cv2.resize(moving_mask, (display_w, display_h))
            moving_bgr = cv2.cvtColor(moving_small, cv2.COLOR_GRAY2BGR)
            
            # 3. Clean Mask (Result of AND operation)
            clean_small = cv2.resize(clean_mask, (display_w, display_h))
            clean_bgr = cv2.cvtColor(clean_small, cv2.COLOR_GRAY2BGR)
            
            # 4. Contour Analysis
            analysis_frame = np.zeros_like(frame)
            
            if contour is not None:
                cv2.drawContours(analysis_frame, [contour], -1, (0, 255, 0), 2)
                cv2.drawContours(analysis_frame, [hull], -1, (255, 0, 0), 2)
                x, y, w, h = bbox
                cv2.rectangle(analysis_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(analysis_frame, "Contour + Hull", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            analysis_small = cv2.resize(analysis_frame, (display_w, display_h))
            
            # Layout:
            # [ Original ] [ Moving Mask ]
            # [ Clean Mask (Skin & Moving) ] [ Final Contour ]
            
            top_row = np.hstack((orig_small, moving_bgr))
            bottom_row = np.hstack((clean_bgr, analysis_small))
            combined = np.vstack((top_row, bottom_row))
            
            # Add text
            cv2.putText(combined, "1. Input", (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(combined, "2. Moving Objects", (490, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(combined, "3. Skin + Moving", (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(combined, "4. Hand", (490, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            
            cv2.imshow("Hand Detection Pipeline (+Background Subtraction)", combined)
            
            if not use_camera:
                print("Synthetic test completed.")
                cv2.waitKey(0)
                break
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if use_camera:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
