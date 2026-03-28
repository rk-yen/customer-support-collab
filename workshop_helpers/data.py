import json
from pathlib import Path

_DATASET_PATH = Path(__file__).with_name("dataset.json")

DATASET = json.loads(_DATASET_PATH.read_text())
