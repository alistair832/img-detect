from pathlib import Path
import runpy

# Single deployment entry point.
# If Streamlit is configured to run the repository root app.py, always launch
# the current Fruits-360 application instead of the old six-class demo.
TARGET_APP = Path(__file__).resolve().parent / "fruits360_full_project" / "app.py"

if not TARGET_APP.exists():
    raise FileNotFoundError(f"Fruits-360 app not found: {TARGET_APP}")

runpy.run_path(str(TARGET_APP), run_name="__main__")
