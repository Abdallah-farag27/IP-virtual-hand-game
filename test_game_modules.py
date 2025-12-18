"""
Test Script: MediaPipe + Custom Game Modules
Demonstrates finger_counter, gesture_detector, and visual_effects working together.
"""

import cv2
import mediapipe as mp

from finger_counter import FingerCounter
from gesture_detector import GestureDetector
from visual_effects import VisualEffects


def main():
    print("=" * 60)
    print("MediaPipe Hand Detection + Custom Game Logic")
    print("=" * 60)
    print("Controls:")
    print("  0 fingers (fist) = SLOW-MO mode (0.5x)")
    print("  1-2 fingers = NORMAL mode (1x)")
    print("  3 fingers = FAST mode (1.5x)")
    print("  4-5 fingers = TURBO mode (2x)")
    print()
    print("Press 'q' to quit, 'r' to reset")
    print("=" * 60)
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Initialize custom modules
    counter = FingerCounter(smoothing_window=5)
    gesture = GestureDetector(stability_frames=8)
    effects = VisualEffects()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip and resize
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (960, 720))
            
            # Apply blur for noise reduction
            frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2RGB)
            
            # Detect hands
            results = hands.process(rgb)
            
            # Default values
            finger_count = 0
            handedness = "Right"
            hand_landmarks = None
            
            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label
                
                # Draw hand using MediaPipe
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Count fingers using our custom module
                finger_count = counter.count_fingers(hand_landmarks, handedness)
                
                # Draw bounding box
                effects.draw_hand_bounding_box(frame, hand_landmarks, 960, 720)
            
            # Detect game mode using our custom module
            mode, speed_mult, score_mult = gesture.detect_mode(finger_count)
            mode_info = gesture.get_mode_info()
            
            # Apply mode overlay
            frame = effects.apply_mode_overlay(frame, mode_info['name'])
            
            # Draw mode indicator
            effects.draw_mode_indicator(frame, mode_info['name'], speed_mult, score_mult)
            
            # Draw finger count
            effects.draw_finger_count(frame, finger_count)
            
            # Draw speed lines for fast modes
            effects.draw_speed_lines(frame, mode_info['name'])
            
            # Show hand type
            if hand_landmarks:
                cv2.putText(frame, f"Hand: {handedness}", (20, 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            else:
                cv2.putText(frame, "No hand detected", (20, 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw column dividers (for game reference)
            for i in range(1, 4):
                x = int(i * 960 / 4)
                cv2.line(frame, (x, 0), (x, 720), (80, 80, 80), 1)
            
            # Display
            cv2.imshow("Hand Game Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                counter.reset()
                gesture.reset()
                print("Reset!")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
