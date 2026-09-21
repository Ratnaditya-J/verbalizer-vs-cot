"""Read-only native attribution check; no model or training dependencies."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from verbalizer_vs_cot.organisms.planted_attribution import evaluate_world


if __name__ == '__main__':
    request = json.load(sys.stdin)
    if set(request) != {'tasks', 'records'}:
        raise ValueError('explicit tasks and original records required')
    print(json.dumps(evaluate_world(request['tasks'], request['records']), allow_nan=False))
