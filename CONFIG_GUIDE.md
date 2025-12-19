# Quick Configuration Guide

## Playing the Game

```bash
python main_game.py
```

**Controls**:

- Move your hand in the **lower half** of the camera view
- `Q` - Quit game
- `R` - Restart (when game over)
- `D` - Toggle debug info (shows hand detection overlay)

---

## Common Configuration Adjustments

### 1. Hand Not Detected?

**Edit `config.py`:**

```python
# Try increasing ROI (more of the frame)
ROI_HEIGHT_RATIO = 0.6  # was 0.5

# Try decreasing minimum area (for smaller hands or farther camera)
MIN_HAND_AREA = 3000  # was 5000

# Enable debug mode to see what's being detected
SHOW_DEBUG_INFO = True
```

### 2. Too Much Noise / False Positives?

**Edit `config.py`:**

```python
# Increase minimum area
MIN_HAND_AREA = 7000  # was 5000

# Decrease maximum area
MAX_HAND_AREA = 40000  # was 50000

# Enable YCrCb for more precise detection
USE_YCRCB = True  # was False
```

### 3. Hand Detection Too Jittery?

**Edit `config.py`:**

```python
# Decrease smoothing alpha (more smoothing)
SMOOTHING_ALPHA = 0.2  # was 0.3

# Enable palm detection (more stable)
USE_PALM_DETECTION = True  # should already be True
```

### 4. Game Too Hard / Too Easy?

**Edit `config.py`:**

```python
# Make easier:
NUM_COLUMNS = 3          # Wider columns
BALL_SPAWN_INTERVAL = 1.5  # Fewer balls
BALL_SPEED = 3           # Slower balls
BOMB_PROBABILITY = 0.1   # Fewer bombs

# Make harder:
NUM_COLUMNS = 4          # Narrower columns
BALL_SPAWN_INTERVAL = 0.8  # More balls
BALL_SPEED = 5           # Faster balls
BOMB_PROBABILITY = 0.3   # More bombs
```

### 5. Different Lighting Conditions?

**Edit `config.py`:**

```python
# For bright lighting:
HSV_LOWER_1 = [0, 15, 80]   # Adjust V (brightness) higher
HSV_UPPER_1 = [20, 255, 255]

# For dim lighting:
HSV_LOWER_1 = [0, 20, 50]   # Adjust V (brightness) lower
HSV_UPPER_1 = [20, 255, 255]

# For challenging lighting (backlighting, shadows):
USE_YCRCB = True  # More robust but slightly slower
```

---

## YCrCb vs HSV - When to Use Each?

### HSV Only (Default - Recommended)

✅ **Use when:**

- Good, even lighting
- Indoor environment
- Static background
- **Faster** (~20% better performance)

**Setting:**

```python
USE_YCRCB = False
```

### HSV + YCrCb (More Robust)

✅ **Use when:**

- Challenging lighting (backlighting, strong shadows)
- Outdoor environment
- Variable lighting conditions
- Need maximum robustness

**Setting:**

```python
USE_YCRCB = True
```

### Performance Impact

- **HSV only**: ~30-35 FPS
- **HSV + YCrCb**: ~25-30 FPS

### Recommendation

**Start with HSV only** (`USE_YCRCB = False`). It's faster and works well in most conditions. Only enable YCrCb if you're experiencing:

- Hand not detected in certain lighting
- Too many false positives from background
- Inconsistent detection as lighting changes

---

## Testing Individual Components

Before playing the full game, test each component:

```bash
# Test hand detection with all improvements
python test_all_steps.py

# Test individual steps
python test_step1_roi.py              # ROI + skin detection
python test_step2_area_filter.py     # Area filtering
python test_step3_smoothing.py       # Temporal smoothing
python test_step4_distance_transform.py  # Palm detection
python test_step5_three_columns.py   # 3-column layout
```

---

## Visualization Panel Size

All test scripts now show panels at **50% size** for better screen fit.

To adjust panel size, edit the test script:

```python
# In any test_stepX.py file, find this line:
scale = 0.5  # 50% size

# Change to:
scale = 0.6  # 60% size (larger)
scale = 0.4  # 40% size (smaller)
```

---

## Full Configuration Reference

See `config.py` for all available settings with detailed comments.

Key sections:

- **Camera Settings** - Resolution, camera index
- **Hand Detection** - ROI, area filtering, smoothing, color ranges
- **Game Settings** - Columns, timer, ball spawning, scoring
- **Display Settings** - Window name, colors, debug mode
- **Performance** - FPS target, visualization scale

---

## Troubleshooting

### Issue: Panels too large, can't see bottom panels

**Solution**: Panels are now 50% size. If still too large, reduce `scale` in test scripts.

### Issue: Hand detected but basket doesn't move

**Solution**: Make sure hand is in the **lower 50%** of the frame (below the yellow ROI line in debug mode).

### Issue: Face being detected as hand

**Solution**: This should not happen with ROI restriction. If it does, decrease `ROI_HEIGHT_RATIO` to 0.4.

### Issue: Background detected as hand

**Solution**:

1. Increase `MIN_HAND_AREA` to filter small regions
2. Decrease `MAX_HAND_AREA` to filter large regions
3. Enable `USE_YCRCB = True` for more precise detection

### Issue: Low FPS (< 20)

**Solution**:

1. Disable YCrCb: `USE_YCRCB = False`
2. Reduce camera resolution in `config.py`
3. Disable debug info: `SHOW_DEBUG_INFO = False`

---

## Quick Start Checklist

1. ✅ Install requirements: `pip install opencv-python numpy`
2. ✅ Test hand detection: `python test_all_steps.py`
3. ✅ Adjust `config.py` if needed (lighting, hand size, etc.)
4. ✅ Play the game: `python main_game.py`
5. ✅ Press `D` in-game to see debug overlay
6. ✅ Adjust settings in `config.py` as needed
7. ✅ Have fun! 🎮

---

## Performance Tips

1. **Start with default settings** - They work well for most cases
2. **Use HSV only** - Enable YCrCb only if needed
3. **Test in your environment** - Run test scripts to see detection quality
4. **Tune incrementally** - Change one setting at a time
5. **Use debug mode** - Press `D` in-game to see what's being detected

---

For complete documentation, see:

- `HAND_DETECTION_README.md` - Full testing guide
- `config.py` - All configuration options with comments
- `walkthrough.md` - Implementation details
