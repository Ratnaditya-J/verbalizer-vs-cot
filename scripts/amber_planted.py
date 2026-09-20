"""JSON subprocess boundary for the prospective planted-rule organism."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from verbalizer_vs_cot.organisms.planted_rule import generate, view, training_record, score, paired_score, specification

request=json.load(sys.stdin);operation=request.pop('operation')
functions={'generate':generate,'view':view,'training_record':training_record,'score':score,
           'paired_score':paired_score,'specification':specification}
if operation not in functions:raise ValueError('unknown planted-rule operation')
print(json.dumps(functions[operation](**request),allow_nan=False))
