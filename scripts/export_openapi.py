"""Write the live OpenAPI document (needs fastapi installed):  python scripts/export_openapi.py docs/openapi.json"""
import json
import sys
from pathlib import Path

# Ensure the repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module_c.api import create_app

path = sys.argv[1] if len(sys.argv) > 1 else "docs/openapi.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(create_app().openapi(), f, indent=2)
print("wrote", path)

