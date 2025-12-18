"""
MediaPipe Hand Detection with Preprocessing
Uses the preprocessing steps from the project before MediaPipe detection.
"""

import cv2
import mediapipe as mp
import numpy as np

# Import your preprocessing module
from preprocess import preprocess_frame, generate_gaussian_kernel


def count_fingers_improved(hand_landmarks, handedness):
    """
    Improved finger counting that handles both left and right hands correctly.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        handedness: "Left" or "Right" hand
        
    Returns:
        int: Number of fingers up (0-5)
    """
    # Landmark indices
    # Thumb: 1-4 (CMC, MCP, IP, TIP)
    # Index: 5-8
    # Middle: 9-12
    # Ring: 13-16
    # Pinky: 17-20
    
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [2, 6, 10, 14, 18]  # Use MCP for thumb (index 2), PIP for others
    
    count = 0
    
    for i, (tip_idx, pip_idx) in enumerate(zip(finger_tips, finger_pips)):
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]
        
        if i == 0:  # Thumb - needs special handling based on hand orientation
            # For right hand (mirrored view): thumb tip should be LEFT of MCP when extended
            # For left hand (mirrored view): thumb tip should be RIGHT of MCP when extended
            thumb_mcp = hand_landmarks.landmark[2]
            thumb_tip = hand_landmarks.landmark[4]
            
            # Check if thumb is extended by comparing x positions
            # In mirrored view: right hand thumb extends left, left hand thumb extends right
            if handedness == "Right":
                # Mirrored right hand: thumb extends to the left (lower x)
                if thumb_tip.x < thumb_mcp.x - 0.04:
                    count += 1
            else:  # Left
                # Mirrored left hand: thumb extends to the right (higher x)
                if thumb_tip.x > thumb_mcp.x + 0.04:
                    count += 1
        else:
            # Other fingers: tip should be above PIP (lower y value) when extended
            if tip.y < pip.y - 0.02:
                count += 1
    
    return count


def main():
    print("=" * 60)
    print("MediaPipe Hand Detection + Preprocessing")
    print("=" * 60)
    print("Using project preprocessing before MediaPipe detection.")
    print("Press 'q' to quit.")
    print("=" * 60)
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
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
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (960, 720))
            
            # ==========================================
            # PREPROCESSING STEP (Using your module)
            # ==========================================
            # Apply Gaussian blur to reduce noise (from your preprocessing)
            gaussian_kernel = generate_gaussian_kernel(ksize=5, sigma=1.0)
            frame_blurred = cv2.filter2D(frame, -1, gaussian_kernel)
            
            # Convert to RGB for MediaPipe (MediaPipe expects RGB)
            rgb = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2RGB)
            
            # ==========================================
            # MEDIAPIPE DETECTION
            # ==========================================
            results = hands.process(rgb)
            
            # Draw landmarks on original frame
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness_info in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # Draw the hand landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Get handedness (Left/Right)
                    handedness = handedness_info.classification[0].label
                    
                    # Get hand center X
                    wrist = hand_landmarks.landmark[0]
                    middle_mcp = hand_landmarks.landmark[9]
                    hand_x = int((wrist.x + middle_mcp.x) / 2 * 960)
                    
                    # Count fingers with improved algorithm
                    finger_count = count_fingers_improved(hand_landmarks, handedness)
                    
                    # Calculate column
                    col = int(hand_x / (960 / 4))
                    col = max(0, min(3, col))
                    
                    # Display info
                    cv2.putText(frame, f"Hand X: {hand_x}", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Fingers: {finger_count}", (20, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Column: {col}", (20, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.putText(frame, f"Hand: {handedness}", (20, 170),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    
                    # Draw bounding box
                    x_coords = [lm.x * 960 for lm in hand_landmarks.landmark]
                    y_coords = [lm.y * 720 for lm in hand_landmarks.landmark]
                    x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
                    y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No hand detected", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show preprocessing indicator
            cv2.putText(frame, "[Preprocessed + MediaPipe]", (680, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Draw column lines
            for i in range(1, 4):
                x = int(i * 960 / 4)
                cv2.line(frame, (x, 0), (x, 720), (80, 80, 80), 1)
            
            cv2.imshow("MediaPipe Hand Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
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
