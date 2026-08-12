# Ensure the repo root is importable so the top-level packages
# (harness/, ecg200/, sleep_edf/) resolve during pytest collection.
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
