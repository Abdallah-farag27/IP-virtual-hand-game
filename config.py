# ============================================================================
# CAMERA SETTINGS
# ============================================================================
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720
CAMERA_INDEX = 0  # 0 for default camera, change if you have multiple cameras

# ============================================================================
# HAND DETECTION SETTINGS
# ============================================================================

# ROI (Region of Interest) Settings
ROI_HEIGHT_RATIO = 0.5  # Fraction of frame height to use (0.5 = lower 50%)
                        # Range: 0.3-0.7
                        # Lower = smaller ROI, faster processing
                        # Higher = larger ROI, more flexibility

# Contour Area Filtering
MIN_HAND_AREA = 5000    # Minimum contour area to consider as hand (pixels)
                        # Increase if detecting too much noise
                        # Decrease if hand not detected when far from camera

MAX_HAND_AREA = 50000   # Maximum contour area to consider as hand (pixels)
                        # Decrease if detecting background as hand
                        # Increase if hand not detected when close to camera

# Temporal Smoothing
SMOOTHING_ALPHA = 0.3   # Exponential smoothing factor (0.0-1.0)
                        # Lower = more smoothing, more lag (e.g., 0.2)
                        # Higher = less smoothing, more responsive (e.g., 0.5)
                        # Recommended: 0.2-0.4

# Detection Method
USE_PALM_DETECTION = True  # True = Distance transform (more stable)
                           # False = Centroid (faster but less stable)

# Skin Color Detection - HSV Color Space
# HSV Range 1 (Reddish skin tones)
HSV_LOWER_1 = [0, 20, 70]      # [Hue, Saturation, Value]
HSV_UPPER_1 = [20, 255, 255]

# HSV Range 2 (Yellowish skin tones)
HSV_LOWER_2 = [170, 20, 70]
HSV_UPPER_2 = [180, 255, 255]

# YCrCb Color Space (Optional - set USE_YCRCB to True to enable)
USE_YCRCB = False       # True = Use both HSV and YCrCb (more robust but slower)
                        # False = Use only HSV (faster, usually sufficient)

# YCrCb Range (used if USE_YCRCB is True)
YCRCB_LOWER = [0, 133, 77]     # [Y, Cr, Cb]
YCRCB_UPPER = [255, 173, 127]

# Morphological Operations
MORPH_KERNEL_SIZE = 5   # Size of morphological kernel (odd number)
                        # Larger = more aggressive noise removal
                        # Smaller = preserves more detail

MORPH_OPEN_ITERATIONS = 2   # Number of opening iterations (removes noise)
MORPH_CLOSE_ITERATIONS = 2  # Number of closing iterations (fills holes)
MORPH_DILATE_ITERATIONS = 1 # Number of dilation iterations (expands hand)

# ============================================================================
# GAME MODES
# ============================================================================
# Multipliers for Speed and Score based on finger count
MODE_SLOW_MULTIPLIER = 0.5   # 0 fingers (Fist)
MODE_NORMAL_MULTIPLIER = 1.0 # 1-2 fingers
MODE_FAST_MULTIPLIER = 1.5   # 3 fibers
MODE_TURBO_MULTIPLIER = 2.0  # 4-5 fingers

# ============================================================================
# GAME SETTINGS
# ============================================================================

# Column Layout
NUM_COLUMNS = 3         # Number of columns (3 recommended, 4 for harder gameplay)

# Column Mapper Settings
ENABLE_COLUMN_SMOOTHING = True  # Smooth column transitions
MAX_COLUMN_JUMP = 1     # Maximum column jump per frame (prevents sudden jumps)

# Game Timer
GAME_DURATION = 90      # Game duration in seconds

# Ball Spawning
BALL_SPAWN_INTERVAL = 1.0   # Time between ball spawns (seconds)
                            # Lower = more balls, harder game
                            # Higher = fewer balls, easier game

MIN_VERTICAL_SPACING = 200  # Minimum vertical distance between balls (pixels)
                            # Prevents overlapping objects

BOMB_PROBABILITY = 0.2  # Probability of spawning a bomb (0.0-1.0)
                        # 0.2 = 20% chance of bomb

BALL_SPEED = 4          # Ball falling speed (pixels per frame)
                        # Higher = faster, harder game

# Scoring
POINTS_PER_BALL = 10    # Points for catching a ball
LIVES_LOST_PER_MISS = 1 # Lives lost for missing a ball
LIVES_LOST_PER_BOMB = 1 # Lives lost for catching a bomb

# Starting Lives
STARTING_LIVES = 3

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

# Window Settings
WINDOW_NAME = "Virtual Hand Basketball Game"
SHOW_FPS = True         # Show FPS counter
SHOW_DEBUG_INFO = False # Show debug information (hand position, etc.)

# Colors (BGR format)
TEXT_COLOR = (255, 255, 255)        # White
ROI_COLOR = (0, 255, 255)           # Yellow
HAND_COLOR = (0, 255, 0)            # Green
COLUMN_DIVIDER_COLOR = (150, 150, 150)  # Gray

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Frame Processing
TARGET_FPS = 30         # Target frames per second
                        # Lower = less CPU usage
                        # Higher = smoother gameplay

# Visualization (for test scripts)
VISUALIZATION_SCALE = 0.5  # Scale factor for visualization windows (0.5 = 50% size)
                           # Adjust if panels are too large for your screen
                           # Range: 0.3-1.0

# ============================================================================
# ADVANCED SETTINGS (Usually don't need to change)
# ============================================================================

# Contour Detection
CONTOUR_RETRIEVAL_MODE = 2  # cv2.RETR_EXTERNAL
CONTOUR_APPROXIMATION = 2   # cv2.CHAIN_APPROX_SIMPLE

# Distance Transform
DISTANCE_TRANSFORM_TYPE = 2  # cv2.DIST_L2
DISTANCE_TRANSFORM_MASK = 5  # Mask size

# Bomb Effect Duration
BOMB_EFFECT_FRAMES = 20  # Number of frames to show explosion effect

# ============================================================================
# NOTES
# ============================================================================
"""
TUNING TIPS:

1. If hand is not detected:
   - Increase ROI_HEIGHT_RATIO (e.g., 0.6)
   - Decrease MIN_HAND_AREA (e.g., 3000)
   - Adjust HSV color ranges for your skin tone and lighting

2. If too much noise/false positives:
   - Decrease MAX_HAND_AREA (e.g., 40000)
   - Increase MIN_HAND_AREA (e.g., 7000)
   - Increase MORPH_OPEN_ITERATIONS (e.g., 3)

3. If hand detection is jittery:
   - Decrease SMOOTHING_ALPHA (e.g., 0.2)
   - Enable USE_PALM_DETECTION

4. If game is too hard/easy:
   - Adjust BALL_SPAWN_INTERVAL
   - Adjust BALL_SPEED
   - Change NUM_COLUMNS (3 = easier, 4 = harder)

5. For different lighting conditions:
   - Test with test_step1_roi.py to see skin detection
   - Adjust HSV ranges accordingly
   - Consider enabling USE_YCRCB for more robust detection

6. YCrCb vs HSV:
   - HSV alone is usually sufficient and faster
   - YCrCb can help in challenging lighting (backlighting, shadows)
   - Using both is more robust but ~20% slower
   - Start with HSV only, enable YCrCb if needed
"""
