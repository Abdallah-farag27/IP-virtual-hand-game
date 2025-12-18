# Virtual Hand Basketball Game - Project Distribution Plan

## 🎮 Project Overview

A virtual hand-controlled basketball game where players move a basket using hand gestures detected via webcam. The game features:

- **Hand tracking** using image processing
- **Finger counting** to control game speed and earn bonus points
- **Ball catching mechanics** with scoring system
- **Real-time visual effects** and UI

---

## 🎯 Core Game Mechanic: Finger-Based Speed Control

**How it works:**

- **1-2 fingers** = Normal speed (1x multiplier)
- **3 fingers** = Fast speed (1.5x multiplier) - basket moves faster, balls fall faster, +50% points
- **4-5 fingers** = Turbo speed (2x multiplier) - everything faster, double points!
- **0 fingers (fist)** = Slow motion (0.5x multiplier) - easier to catch, but half points

This adds strategy: players must balance speed vs. accuracy!

---

## 👥 Team Member Assignments

### **Person 1: Hassan - Image Preprocessing & Skin Detection**

#### 📁 Files to Create:

- `preprocess.py`
- `skin_detector.py`

#### 🔧 Image Processing Tasks:

**1. Image Preprocessing (`preprocess.py`)**

- Resize frame to standard size (960x720)
- Flip frame horizontally (mirror effect)
- Apply **Gaussian Blur** to reduce noise
- Convert BGR to **HSV color space**
- Apply **Histogram Equalization** for better lighting
- **Brightness/Contrast adjustment** for different environments

**2. Skin Detection (`skin_detector.py`)**

- Define HSV range for skin color detection
- Create binary mask using `cv2.inRange()`
- Apply **morphological operations**:
  - **Erosion** to remove noise
  - **Dilation** to fill gaps
  - **Opening** (erosion + dilation) to clean mask
  - **Closing** (dilation + erosion) to fill holes
- Return clean binary mask of hand region

#### 🎮 Game Logic Contribution:

- Integrate preprocessing into main game loop
- Ensure consistent frame quality for other modules
- Handle edge cases (no camera, poor lighting)

**Estimated Work:** ~150-180 lines of code

---

### **Person 2: Wafa - Hand Contour Detection & Centroid Calculation**

#### 📁 Files to Create:

- `hand_contour.py`
- `centroid_calculator.py`

#### 🔧 Image Processing Tasks:

**1. Hand Contour Detection (`hand_contour.py`)**

- Find all contours in binary mask using `cv2.findContours()`
- Filter contours by area (remove small noise)
- Select **largest contour** as hand
- Calculate **bounding rectangle** around hand
- Extract **convex hull** of hand contour
- Return hand contour and bounding box

**2. Centroid Calculation (`centroid_calculator.py`)**

- Calculate **center of mass** using image moments (`cv2.moments()`)
- Implement **smoothing filter** (moving average over last 5 frames)
- Handle missing detections (use previous position)
- Return smoothed (x, y) centroid coordinates
- Optionally draw centroid on frame for debugging

#### 🎮 Game Logic Contribution:

- Provide hand position to column mapper
- Ensure smooth basket movement (no jitter)
- Handle hand entering/leaving frame

**Estimated Work:** ~120-150 lines of code

---

### **Person 3: Fararg & Tony (Part 1) - Finger Counting & Gesture Recognition**

#### 📁 Files to Create:

- `finger_counter.py`
- `gesture_detector.py`

#### 🔧 Image Processing Tasks:

**1. Finger Counting (`finger_counter.py`)**

- Use **convex hull** from hand contour
- Calculate **convexity defects** using `cv2.convexityDefects()`
- Count defects that represent gaps between fingers
- Apply filtering:
  - Minimum depth threshold
  - Angle threshold (to avoid false positives)
- Return number of fingers (0-5)
- Implement **temporal smoothing** (average over 3-5 frames)

**2. Gesture Recognition (`gesture_detector.py`)**

- Map finger count to game modes:
  - **0 fingers (fist)** → Slow-mo mode (0.5x speed, 0.5x points)
  - **1-2 fingers** → Normal mode (1x speed, 1x points)
  - **3 fingers** → Fast mode (1.5x speed, 1.5x points)
  - **4-5 fingers** → Turbo mode (2x speed, 2x points)
- Implement **gesture stability** (require gesture for 10 frames before switching)
- Return current game mode and multiplier

#### 🎮 Game Logic Contribution:

- Update ball falling speed based on finger count
- Apply score multiplier to caught balls
- Display current mode on screen
- Add visual indicator (color change) for different modes

**Estimated Work:** ~180-220 lines of code (split between 2 people)

---

### **Person 4: Tony (Part 2) - Visual Effects & Integration**

#### 📁 Files to Create/Update:

- `visual_effects.py` (new)
- Update `renderer.py` (existing)
- Update `main_game.py` (integration)

#### 🔧 Image Processing Tasks:

**1. Visual Effects (`visual_effects.py`)**

- **Hand outline drawing** (draw contour on frame)
- **Finger count visualization** (show number on screen)
- **Mode indicator effects**:
  - Slow-mo: Blue tint overlay
  - Normal: No effect
  - Fast: Yellow glow
  - Turbo: Red/orange glow with particles
- **Speed lines effect** when in fast/turbo mode
- **Smooth color transitions** between modes

**2. Enhanced Rendering (update `renderer.py`)**

- Add mode indicator panel
- Display current multiplier
- Show finger count with visual representation
- Add color-coded basket (changes with mode)
- Particle effects for turbo mode

#### 🎮 Game Logic Contribution:

- **Integrate all modules** in `main_game.py`:
  ```
  Frame → Preprocess → Skin Detection → Contour Detection
       → Centroid → Finger Count → Mode Detection → Update Game
  ```
- Apply speed multiplier to ball physics
- Apply score multiplier when catching balls
- Update UI to show current mode and multiplier
- Handle mode transitions smoothly
- Add FPS counter for performance monitoring

**Estimated Work:** ~200-250 lines of code

---

## 📊 Detailed Module Specifications

### Module 1: `preprocess.py` (Hassan)

```python
def preprocess_frame(frame):
    """
    Preprocess camera frame for hand detection

    Steps:
    1. Resize to 960x720
    2. Flip horizontally
    3. Apply Gaussian blur (kernel size 5x5)
    4. Convert BGR to HSV
    5. Apply histogram equalization on V channel

    Returns: preprocessed_frame (HSV)
    """
    pass
```

### Module 2: `skin_detector.py` (Hassan)

```python
def detect_skin(hsv_frame):
    """
    Create binary mask of skin regions

    Steps:
    1. Define HSV range for skin (H: 0-20, S: 20-255, V: 70-255)
    2. Apply cv2.inRange()
    3. Morphological operations:
       - Erosion (3x3 kernel, 2 iterations)
       - Dilation (5x5 kernel, 2 iterations)

    Returns: binary_mask (0 or 255)
    """
    pass
```

### Module 3: `hand_contour.py` (Wafa)

```python
def find_hand_contour(binary_mask):
    """
    Find largest hand contour

    Steps:
    1. Find all contours
    2. Filter by area (min 5000 pixels)
    3. Select largest contour
    4. Calculate bounding rectangle
    5. Calculate convex hull

    Returns: contour, bounding_box, convex_hull
    """
    pass
```

### Module 4: `centroid_calculator.py` (Wafa)

```python
def calculate_centroid(contour, smooth=True):
    """
    Calculate hand centroid with smoothing

    Steps:
    1. Calculate moments using cv2.moments()
    2. Calculate center: cx = M10/M00, cy = M01/M00
    3. Apply moving average filter (window size 5)
    4. Handle None/invalid cases

    Returns: (x, y) centroid coordinates
    """
    pass
```

### Module 5: `finger_counter.py` (Fararg & Tony)

```python
def count_fingers(contour, convex_hull):
    """
    Count extended fingers using convexity defects

    Steps:
    1. Calculate convexity defects
    2. Filter defects:
       - Depth > 20 pixels
       - Angle < 90 degrees
    3. Count valid defects
    4. Fingers = defects + 1 (usually)
    5. Apply temporal smoothing

    Returns: finger_count (0-5)
    """
    pass
```

### Module 6: `gesture_detector.py` (Fararg & Tony)

```python
def detect_game_mode(finger_count):
    """
    Map finger count to game mode

    Modes:
    - 0 fingers: SLOW_MO (0.5x speed, 0.5x points)
    - 1-2 fingers: NORMAL (1x speed, 1x points)
    - 3 fingers: FAST (1.5x speed, 1.5x points)
    - 4-5 fingers: TURBO (2x speed, 2x points)

    Returns: mode_name, speed_multiplier, score_multiplier
    """
    pass
```

### Module 7: `visual_effects.py` (Tony)

```python
def draw_hand_overlay(frame, contour, finger_count, mode):
    """
    Draw visual effects on frame

    Effects:
    1. Draw hand contour (green outline)
    2. Draw finger count number
    3. Apply mode-specific overlay:
       - SLOW_MO: blue tint
       - FAST: yellow glow
       - TURBO: red glow + particles

    Returns: frame with effects
    """
    pass
```

---

## 🔄 Integration Flow

```
┌─────────────────┐
│  Camera Frame   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Preprocess    │ ← Hassan
│  (resize, blur, │
│   HSV convert)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Skin Detection │ ← Hassan
│ (binary mask +  │
│  morphology)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Find Contour &  │ ← Wafa
│  Convex Hull    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Calculate     │ ← Wafa
│    Centroid     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Count Fingers  │ ← Fararg & Tony
│ (convexity def) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Detect Game Mode│ ← Fararg & Tony
│  (map to speed) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Update Game    │ ← Tony
│ (balls, score,  │
│   rendering)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Display Frame   │
└─────────────────┘
```

---

## 🎮 Game Logic Updates (Each Person's Part)

### Hassan's Game Logic:

```python
# In main_game.py
preprocessed = preprocess_frame(frame)
skin_mask = detect_skin(preprocessed)
```

### Wafa's Game Logic:

```python
# In main_game.py
contour, bbox, hull = find_hand_contour(skin_mask)
if contour is not None:
    centroid = calculate_centroid(contour)
    basket_col = mapper.map_to_column(centroid['x'], 960)
```

### Fararg & Tony's Game Logic:

```python
# In main_game.py
finger_count = count_fingers(contour, hull)
mode, speed_mult, score_mult = detect_game_mode(finger_count)

# Update ball speed
for ball in balls:
    ball['speed'] = base_speed * speed_mult

# Update score when catching
if ball_caught:
    score += 10 * score_mult
```

### Tony's Integration:

```python
# In main_game.py - complete game loop
while True:
    ret, frame = cap.read()

    # Hassan's modules
    preprocessed = preprocess_frame(frame)
    skin_mask = detect_skin(preprocessed)

    # Wafa's modules
    contour, bbox, hull = find_hand_contour(skin_mask)
    centroid = calculate_centroid(contour)

    # Fararg & Tony's modules
    finger_count = count_fingers(contour, hull)
    mode, speed_mult, score_mult = detect_game_mode(finger_count)

    # Update game
    basket_col = mapper.map_to_column(centroid['x'], 960)
    update_balls(balls, speed_mult)
    check_collisions(balls, basket_col, score_mult)

    # Tony's rendering
    draw_hand_overlay(frame, contour, finger_count, mode)
    renderer.render_ui(frame, balls, basket_col, score, lives, time_left, mode)

    cv2.imshow('Game', frame)
```

---

## 📋 Testing Checklist

### Hassan's Tests:

- [ ] Preprocessing works in different lighting
- [ ] Skin detection works for different skin tones
- [ ] Morphological operations clean the mask properly
- [ ] HSV conversion is correct

### Wafa's Tests:

- [ ] Contour detection finds hand reliably
- [ ] Centroid is calculated correctly
- [ ] Smoothing reduces jitter
- [ ] Handles hand leaving/entering frame

### Fararg & Tony's Tests:

- [ ] Finger counting is accurate (0-5)
- [ ] Gesture detection is stable (no flickering)
- [ ] Mode transitions are smooth
- [ ] Multipliers apply correctly

### Tony's Integration Tests:

- [ ] All modules work together
- [ ] Game runs at 30+ FPS
- [ ] No crashes or errors
- [ ] Visual effects look good
- [ ] Score multiplier works correctly

---

## 🚀 Development Timeline

### Week 1: Individual Module Development

- Each person creates their modules
- Test independently with sample images/videos
- Document any issues

### Week 2: Integration

- Tony integrates all modules into `main_game.py`
- Team tests together
- Fix bugs and edge cases

### Week 3: Polish & Testing

- Add visual effects
- Test with different people/lighting
- Fine-tune parameters (HSV ranges, thresholds)
- Create demo video

---

## 📊 Workload Summary

| Person            | Image Processing Tasks                    | Game Logic Tasks           | Lines of Code |
| ----------------- | ----------------------------------------- | -------------------------- | ------------- |
| **Hassan**        | Preprocessing, Skin Detection, Morphology | Frame preparation pipeline | ~150-180      |
| **Wafa**          | Contour Detection, Centroid, Smoothing    | Basket position mapping    | ~120-150      |
| **Fararg & Tony** | Finger Counting, Convexity Defects        | Speed/score multipliers    | ~180-220      |
| **Tony**          | Visual Effects, Integration               | Complete game loop         | ~200-250      |

**Total:** ~650-800 lines of code

---

## 🎯 Success Criteria

✅ Hand detection works in real-time (30+ FPS)  
✅ Finger counting is accurate and stable  
✅ Game speed changes based on finger count  
✅ Score multiplier works correctly  
✅ Visual effects enhance gameplay  
✅ No crashes or major bugs  
✅ Fun to play!

---

## 💡 Tips for Success

1. **Test early, test often** - Don't wait until integration to test
2. **Use print statements** - Debug by printing values
3. **Save test images** - Capture frames for offline testing
4. **Communicate** - Share progress and issues with team
5. **Parameter tuning** - HSV ranges and thresholds need adjustment
6. **Handle edge cases** - What if no hand detected? What if multiple hands?

---

## 📞 Communication Plan

- **Daily standups** (5 min): Share progress and blockers
- **Code reviews**: Review each other's code before integration
- **Shared testing**: Test together to find edge cases
- **Documentation**: Comment your code clearly

---

## 🎉 Bonus Features (If Time Permits)

- **Calibration mode**: Let user adjust HSV ranges for their skin tone
- **Multiple difficulty levels**: Adjust ball speed and spawn rate
- **Sound effects**: Add audio feedback
- **High score system**: Save and display best scores
- **Replay system**: Record and playback gameplay

---

Good luck team! 🚀 Let's build an amazing virtual hand basketball game!
