# Configuration file for Virtual Hand Basketball Game

# ============================================
# DISPLAY SETTINGS
# ============================================
FRAME_WIDTH = 960
FRAME_HEIGHT = 720
FPS_TARGET = 30

# ============================================
# GAME SETTINGS
# ============================================
NUM_COLUMNS = 4
GAME_DURATION = 90  # seconds
INITIAL_LIVES = 3

# Ball spawning
BALL_SPAWN_INTERVAL = 1.0  # seconds
BASE_BALL_SPEED = 4  # pixels per frame
BOMB_PROBABILITY = 0.2  # 20% chance

# Scoring
POINTS_PER_BALL = 10
POINTS_LOST_PER_MISS = 0  # or set to negative value to deduct points

# ============================================
# IMAGE PROCESSING SETTINGS
# ============================================

# Preprocessing
GAUSSIAN_BLUR_KERNEL = (5, 5)
HISTOGRAM_EQUALIZATION = True

# Skin Detection (HSV color space)
# Adjust these values based on your skin tone and lighting
SKIN_HSV_LOWER = (0, 20, 70)    # (Hue, Saturation, Value)
SKIN_HSV_UPPER = (20, 255, 255)

# Alternative skin ranges for different skin tones (uncomment to use)
# SKIN_HSV_LOWER = (0, 30, 60)   # Lighter skin
# SKIN_HSV_UPPER = (25, 255, 255)

# SKIN_HSV_LOWER = (0, 40, 50)   # Darker skin
# SKIN_HSV_UPPER = (30, 255, 255)

# Morphological Operations
EROSION_KERNEL_SIZE = (3, 3)
EROSION_ITERATIONS = 2
DILATION_KERNEL_SIZE = (5, 5)
DILATION_ITERATIONS = 2

# Hand Detection
MIN_HAND_AREA = 5000  # minimum pixels for valid hand contour
MAX_HAND_AREA = 200000  # maximum pixels (to filter out full body)

# Centroid Smoothing
CENTROID_SMOOTH_WINDOW = 5  # number of frames to average

# Finger Counting
MIN_DEFECT_DEPTH = 20  # minimum depth for valid convexity defect
MAX_DEFECT_ANGLE = 90  # maximum angle in degrees
FINGER_COUNT_SMOOTH_WINDOW = 3  # frames to average for stable count

# ============================================
# GAME MODE SETTINGS (Finger-based)
# ============================================

# Mode definitions: (speed_multiplier, score_multiplier)
GAME_MODES = {
    'SLOW_MO': {
        'finger_range': (0, 0),      # 0 fingers (fist)
        'speed_mult': 0.5,
        'score_mult': 0.5,
        'color': (255, 100, 100),    # Blue tint (BGR)
        'name': 'SLOW-MO'
    },
    'NORMAL': {
        'finger_range': (1, 2),      # 1-2 fingers
        'speed_mult': 1.0,
        'score_mult': 1.0,
        'color': (255, 255, 255),    # White (BGR)
        'name': 'NORMAL'
    },
    'FAST': {
        'finger_range': (3, 3),      # 3 fingers
        'speed_mult': 1.5,
        'score_mult': 1.5,
        'color': (0, 255, 255),      # Yellow (BGR)
        'name': 'FAST'
    },
    'TURBO': {
        'finger_range': (4, 5),      # 4-5 fingers
        'speed_mult': 2.0,
        'score_mult': 2.0,
        'color': (0, 100, 255),      # Orange/Red (BGR)
        'name': 'TURBO'
    }
}

# Mode transition stability (frames to hold gesture before switching)
MODE_STABILITY_FRAMES = 10

# ============================================
# VISUAL EFFECTS SETTINGS
# ============================================

# Hand visualization
DRAW_HAND_CONTOUR = True
HAND_CONTOUR_COLOR = (0, 255, 0)  # Green (BGR)
HAND_CONTOUR_THICKNESS = 2

DRAW_CONVEX_HULL = True
HULL_COLOR = (0, 255, 255)  # Yellow (BGR)
HULL_THICKNESS = 2

DRAW_CENTROID = True
CENTROID_COLOR = (255, 0, 255)  # Magenta (BGR)
CENTROID_RADIUS = 8

# Mode effects
MODE_OVERLAY_ALPHA = 0.2  # transparency for mode color overlay
SHOW_MODE_INDICATOR = True
SHOW_FINGER_COUNT = True

# ============================================
# DEBUG SETTINGS
# ============================================
DEBUG_MODE = False  # Set to True to show debug information
SHOW_FPS = True
SHOW_BINARY_MASK = False  # Show skin detection mask in separate window
SHOW_CONTOURS = True

# ============================================
# CALIBRATION MODE
# ============================================
ENABLE_CALIBRATION = False  # Set to True to enable HSV calibration on startup
