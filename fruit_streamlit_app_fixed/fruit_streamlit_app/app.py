from pathlib import Path

# Compatibility entrypoint: if Streamlit Cloud is still configured to use this
# older nested file, execute the new root live-camera app instead.
ROOT_APP = Path(__file__).resolve().parents[2] / "app.py"
exec(compile(ROOT_APP.read_text(encoding="utf-8"), str(ROOT_APP), "exec"))
