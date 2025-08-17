from pathlib import Path

ANNOTATIONS_FILE = Path(__file__).parent / "UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = Path(__file__).parent / "UrbanSound8K/audio"

assert ANNOTATIONS_FILE.exists(), f"Annotations file {ANNOTATIONS_FILE} does not exist."
assert AUDIO_DIR.exists(), f"Audio directory {AUDIO_DIR} does not exist."
