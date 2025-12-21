# Virtual Hand Basketball Game

A computer vision-based game where you control a basket using your hand movements to catch falling balls and avoid bombs.

## How to Play

1.  **Run the Game**: Launch the application.
2.  **Control**: Move your hand in the **LOWER half** of the camera view.
    *   The basket moves horizontally with your hand.
    *   The game detects your hand using skin color segmentation and contours.
3.  **Objective**:
    *   **Catch Balls**: Align the basket with falling basketballs to score **10 points**.
    *   **Avoid Bombs**: Dodge the bombs! Catching one costs you a **life**.
    *   **Missed Balls**: Letting a regular ball pass also costs a **life**.
4.  **Game Over**: The game ends when you run out of lives or the timer expires.

## Setup Instructions

### 1. Create a Virtual Environment (Recommended)

Open your terminal in the project folder and run:

**Windows (Git Bash / bash):**
```bash
python -m venv venv
source venv/Scripts/activate
```

**Windows (CMD / PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate.bat
```

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

*Note: Ensure you have a working webcam connected.*

## Running the Game

To start the main game:

```bash
python main_game.py
```

## Running Tests

We include several test scripts to visualize and debug different stages of the image processing pipeline:

*   **`test_step1_roi.py`**: Visualizes the **Region of Interest (ROI)** selection and skin color segmentation.
*   **`test_step2_area_filter.py`**: Tests contour detection and filters out noise based on area.
*   **`test_step3_smoothing.py`**: Demonstrates the temporal smoothing applied to hand movements to reduce jitter.
*   **`test_step4_distance_transform.py`**: Visualizes **palm detection** using the distance transform method (more stable than centroid).
*   **`test_step5_three_columns.py`**: Verifies how the continuous hand position is mapped to the discrete game columns.
*   **`test_all_steps.py`**: Runs a comprehensive visualization of the entire pipeline simultaneously.

## Tuning Tips

If you experience issues with detection or gameplay, adjust the settings in `config.py`.

**1. Hand Not Detected?**
*   Increase `ROI_HEIGHT_RATIO` (e.g., 0.6) to look at a larger portion of the screen.
*   Decrease `MIN_HAND_AREA` (e.g., 3000) if your hand is smaller or further away.
*   **Crucial**: Adjust `HSV_LOWER_1` / `HSV_UPPER_1` in `config.py` to match your specific skin tone and lighting.

**2. Too Much Noise / False Positives?**
*   Increase `MIN_HAND_AREA` (e.g., 7000).
*   Increase `MORPH_OPEN_ITERATIONS` (e.g., 3) to remove small noise specs.

**3. Jittery Movement?**
*   Decrease `SMOOTHING_ALPHA` (e.g., 0.2) for smoother but slightly laggier movement.
*   Ensure `USE_PALM_DETECTION` is set to `True` for better stability.

**4. Lighting Issues?**
*   Use `test_step1_roi.py` to tune your HSV values.
*   Try enabling `USE_YCRCB = True` in `config.py` for more robust detection in difficult lighting (slightly slower).

**5. Gameplay Difficulty**
*   Change `BALL_SPAWN_INTERVAL` to make balls appear faster or slower.
*   Adjust `BALL_SPEED` to change how fast they fall.
*   Modify `NUM_COLUMNS` (3 is standard, 4 is harder).
