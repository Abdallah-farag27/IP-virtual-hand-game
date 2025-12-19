# Hand Detection Improvements - Testing Guide

## Overview

This project implements robust hand detection for a virtual hand basketball game using **classical image processing techniques only** (no neural networks or deep learning).

## Implemented Improvements

### ✅ Step 1: ROI Restriction + Skin Detection

- **File**: `hand_detector.py`
- **Test**: `test_step1_roi.py`
- **Features**:
  - Detection limited to lower 50% of frame
  - Eliminates face detection
  - HSV-based skin color segmentation
  - Morphological operations for noise reduction

### ✅ Step 2: Contour Area Filtering

- **File**: `hand_detector.py` (integrated)
- **Test**: `test_step2_area_filter.py`
- **Features**:
  - Filters contours by area (5000-50000 pixels)
  - Removes small noise and large backgrounds
  - Color-coded visualization of rejected/accepted contours

### ✅ Step 3: Temporal Smoothing

- **File**: `hand_detector.py` (integrated)
- **Test**: `test_step3_smoothing.py`
- **Features**:
  - Exponential moving average (α=0.3)
  - Reduces jitter and sudden jumps
  - Adjustable smoothing factor
  - Side-by-side comparison with raw detection

### ✅ Step 4: Distance Transform Palm Detection

- **File**: `hand_detector.py` (integrated)
- **Test**: `test_step4_distance_transform.py`
- **Features**:
  - Finds palm center (most stable part of hand)
  - More stable than centroid when fingers move
  - Heatmap visualization
  - Comparison with centroid method

### ✅ Step 5: 3-Column Layout

- **Files**: `column_mapper.py`, `renderer.py`
- **Test**: `test_step5_three_columns.py`
- **Features**:
  - Reduced from 4 to 3 columns
  - Wider columns (320px vs 240px)
  - Less sensitivity to hand jitter
  - Easier gameplay control

## Testing Instructions

### Quick Start - Test All Improvements Together

```bash
python test_all_steps.py
```

This comprehensive test shows all improvements working together with:

- Live camera feed with hand detection
- 3-column layout with basket
- Real-time FPS display
- Feature status panel
- Interactive controls

**Controls**:

- `Q` - Quit
- `R` - Reset
- `P` - Toggle between palm detection and centroid
- `+` - Increase smoothing
- `-` - Decrease smoothing

---

### Individual Step Testing

Test each improvement separately to see the progress:

#### Step 1: ROI Restriction

```bash
python test_step1_roi.py
```

**What to observe**:

- 4-panel view showing original frame, ROI boundary, skin mask, and detection result
- Your face (in upper half) should be ignored
- Only hand in lower half is detected

**Controls**:

- `S` - Toggle smoothing on/off
- `R` - Reset

---

#### Step 2: Contour Area Filtering

```bash
python test_step2_area_filter.py
```

**What to observe**:

- Color-coded contours:
  - **Red**: Too small (noise)
  - **Blue**: Too large (background)
  - **Yellow**: Valid size
  - **Green**: Selected hand
- Statistics panel showing contour counts and areas

---

#### Step 3: Temporal Smoothing

```bash
python test_step3_smoothing.py
```

**What to observe**:

- Side-by-side comparison of raw vs smoothed detection
- Position history graph showing both methods
- Jitter reduction percentage
- Move your hand quickly to see the smoothing effect!

**Controls**:

- `+` - Increase smoothing (higher α)
- `-` - Decrease smoothing (lower α)

---

#### Step 4: Distance Transform

```bash
python test_step4_distance_transform.py
```

**What to observe**:

- Comparison of centroid vs palm center detection
- Distance transform heatmap (red = far from edge, blue = near edge)
- Palm radius visualization
- Try moving your fingers while keeping palm still!

---

#### Step 5: 3-Column Layout

```bash
python test_step5_three_columns.py
```

**What to observe**:

- Side-by-side comparison of 3-column vs 4-column layouts
- Column width comparison bars
- Advantages of wider columns

---

## File Structure

```
IP-virtual-hand-game/
├── hand_detector.py              # Main hand detection module (all improvements)
├── column_mapper.py              # Maps hand position to columns (3 columns)
├── renderer.py                   # Game rendering (3 columns)
├── game_timer.py                 # Game timer
├── main_game.py                  # Main game (to be updated)
├── test_tony.py                  # Original test
│
├── test_step1_roi.py             # Test Step 1: ROI + Skin Detection
├── test_step2_area_filter.py    # Test Step 2: Area Filtering
├── test_step3_smoothing.py      # Test Step 3: Temporal Smoothing
├── test_step4_distance_transform.py  # Test Step 4: Palm Detection
├── test_step5_three_columns.py  # Test Step 5: 3 Columns
└── test_all_steps.py            # Comprehensive test (all improvements)
```

## Requirements

```bash
pip install opencv-python numpy
```

## Performance

All improvements maintain real-time performance:

- **Target FPS**: 30+ fps
- **Tested on**: Standard webcam (720p)
- **Processing time**: ~20-30ms per frame

## Next Steps

1. **Test each step individually** to understand each improvement
2. **Run `test_all_steps.py`** to see everything working together
3. **Integrate into `main_game.py`** for the actual game
4. **Tune parameters** based on your lighting conditions:
   - Skin color thresholds in `hand_detector.py`
   - Area filtering range (min_area, max_area)
   - Smoothing factor (smoothing_alpha)
   - ROI height ratio

## Tips for Best Results

1. **Lighting**: Use good, even lighting. Avoid strong backlighting.
2. **Background**: Static backgrounds work best. Avoid skin-colored walls.
3. **Hand Position**: Keep hand in lower 50% of frame.
4. **Hand Size**: Ensure hand fills reasonable portion of ROI (not too close/far).
5. **Movement**: Smooth, deliberate movements work better than jerky motions.

## Troubleshooting

### Hand not detected

- Check lighting conditions
- Adjust skin color thresholds in `hand_detector.py`
- Ensure hand is in lower 50% of frame
- Try adjusting area filtering range

### Too much jitter

- Increase smoothing (lower α value)
- Use palm detection instead of centroid
- Check for background interference

### False positives

- Adjust area filtering to be more restrictive
- Check ROI is properly set
- Ensure background is not skin-colored

## Classical Image Processing Techniques Used

This project demonstrates the following classical CV techniques:

1. **Color Space Conversion**: BGR → HSV
2. **Color Thresholding**: HSV range-based skin detection
3. **Morphological Operations**: Opening, closing, dilation
4. **Contour Detection**: Finding connected components
5. **Contour Filtering**: Area-based filtering
6. **Moments**: Centroid calculation
7. **Distance Transform**: Finding palm center
8. **Temporal Filtering**: Exponential moving average
9. **ROI Processing**: Spatial restriction

**No neural networks, no deep learning, no pre-trained models!**

## Credits

Developed for educational purposes to demonstrate robust hand tracking using only classical image processing techniques.
