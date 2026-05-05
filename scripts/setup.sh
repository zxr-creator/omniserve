#!/usr/bin/env bash
# Download DeepSeek-R1-Distill-Llama-8B, patch its rope_scaling for the
# transformers==4.37.2 OmniServe pin, and point lserve_e2e.sh at the local copy.
#
# Usage:
#   bash setup_r1_distill.sh                       # defaults to /tmp/models
#   MODEL_ROOT=/some/big/disk bash setup_r1_distill.sh
set -euo pipefail

MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_ROOT="${MODEL_ROOT:-/tmp/models}"
MODEL_DIR="${MODEL_ROOT}/$(basename "${MODEL_ID}")"
LSERVE_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lserve_e2e.sh"

echo "[1/4] Downloading ${MODEL_ID} -> ${MODEL_DIR}"
mkdir -p "${MODEL_ROOT}"
python - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download(
    "${MODEL_ID}",
    local_dir="${MODEL_DIR}",
)
print("downloaded to:", path)
PY

CFG="${MODEL_DIR}/config.json"
if [[ ! -f "${CFG}" ]]; then
    echo "ERROR: ${CFG} not found after download" >&2
    exit 1
fi

echo "[2/4] Patching rope_scaling in ${CFG}"
cp -n "${CFG}" "${CFG}.bak" || true
python - <<PY
import json, pathlib
p = pathlib.Path("${CFG}")
cfg = json.loads(p.read_text())
print("  OLD:", cfg.get("rope_scaling"))
cfg["rope_scaling"] = {"type": "linear", "factor": 8.0}
p.write_text(json.dumps(cfg, indent=2))
print("  NEW:", cfg["rope_scaling"])
PY

echo "[3/4] Verifying config loads under current transformers"
python -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('${MODEL_DIR}')
print('  OK, rope_scaling =', c.rope_scaling)
"

echo "[4/4] Updating model_path in ${LSERVE_SH}"
if [[ -f "${LSERVE_SH}" ]]; then
    cp -n "${LSERVE_SH}" "${LSERVE_SH}.bak" || true
    sed -i "s|^model_path=.*|model_path=\"${MODEL_DIR}\"|" "${LSERVE_SH}"
    echo "  $(grep '^model_path=' "${LSERVE_SH}")"
else
    echo "  (skipped: ${LSERVE_SH} not found)"
fi

echo
echo "Done. Run:  bash ${LSERVE_SH}"
