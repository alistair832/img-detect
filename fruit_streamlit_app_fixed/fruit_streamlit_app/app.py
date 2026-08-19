from pathlib import Path

# Compatibility entrypoint: if Streamlit Cloud is still configured to use this
# older nested file, execute the new root app while making __file__ point to the
# root app. This keeps all relative paths (such as legacy_model.py) correct.
ROOT_APP = Path(__file__).resolve().parents[2] / "app.py"

globals()["__file__"] = str(ROOT_APP)
exec(compile(ROOT_APP.read_text(encoding="utf-8"), str(ROOT_APP), "exec"), globals())
