#!/usr/bin/env bash
# Pod bootstrap: unpack code, install, preflight. Run in /workspace.
set -euo pipefail
cd /workspace
tar xzf code.tgz
python -m pip install -q -e "sieve-audit[runner,judges]"
python -m pip install -q -e "verbalizer-vs-cot"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA"
print("[pod] CUDA:", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")
import sieve_audit, verbalizer_vs_cot
from sieve_audit.adapters import verbalizer
print("[pod] sieve_audit", sieve_audit.__version__, "verbalizer adapter OK")
PY
echo "[pod] bootstrap done"
