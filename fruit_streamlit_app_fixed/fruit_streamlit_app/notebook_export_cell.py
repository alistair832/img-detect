# Add this cell at the END of your existing notebook after training finishes.
import json
from pathlib import Path

Path("models").mkdir(exist_ok=True)
model.save("models/fruit_resnet50.keras")

with open("models/class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

print("Saved models/fruit_resnet50.keras")
print("Saved models/class_names.json")
