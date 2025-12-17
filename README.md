# Run Instructions — Virtual Hand Basketball Game

This file explains only how to create a virtual environment, install required libraries, and run the game.

1. Create a virtual environment (from project root):

```bash
python -m venv venv
```

2. Activate the virtual environment (Git Bash / bash on Windows):

```bash
source venv/Scripts/activate
```

(If you use CMD on Windows run: `venv\Scripts\activate.bat`)

3. Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

4. Run the test game using your webcam:

```bash
python test_tony.py
```

5. When finished, deactivate the environment:

```bash
deactivate
```

One-line description of project files:

- `column_mapper.py` : Map hand x-coordinate to a column index with optional smoothing.
- `renderer.py` : Draw nets, balls, UI, and game-over screen (all rendering code).
- `game_timer.py` : Countdown timer and helpers (start/pause/reset).
- `test_tony.py` : Camera-backed test harness — runs the game loop for testing.
- `requirements.txt` : Lists Python packages required to run the game.

That's it — follow the commands above to set up a venv, install libs, and run the game.
