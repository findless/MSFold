#!/bin/bash
# MSFold release smoke test
# Run on a GPU node with msfold conda environment
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "MSFold Release Smoke Test"
echo "============================================"
echo ""

# --- Check environment ---
echo "[1/5] Checking environment..."
python -c "import torch; print('  PyTorch', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"
python -c "from esm.models.esm3 import ESM3; print('  ESM3 importable')"
python -c "from msfold.models.esm3 import load_esm3_client; print('  msfold importable')"
echo "  OK"
echo ""

# --- Quick test (smoke) ---
echo "[2/5] Quick smoke test (4 replicas, 2 steps)..."
rm -rf outputs/test_quick
python scripts/run_msfold_sample.py \
    --fasta examples/example.fasta \
    --config configs/default.yaml \
    --out outputs/test_quick \
    --device cuda \
    --seed 42
echo "  OK (see outputs/test_quick/samples.csv)"
echo ""

# --- Rank SLL ---
echo "[3/5] SLL ranking..."
python scripts/rank_sll.py \
    --sample_dir outputs/test_quick \
    --out outputs/test_quick/sll_ranking.csv
echo "  OK"
echo ""

# --- Check outputs ---
echo "[4/5] Output summary:"
echo "  samples.csv: $(wc -l < outputs/test_quick/samples.csv) lines (incl header)"
echo "  sll_ranking.csv: $(wc -l < outputs/test_quick/sll_ranking.csv) lines (incl header)"
echo "  PDB files: $(ls outputs/test_quick/*.pdb 2>/dev/null | wc -l)"
cat outputs/test_quick/run_metadata.json
echo ""

# --- Full test instructions ---
echo "[5/5] Full default test (40 replicas, 500 steps):"
echo "  Run manually:"
echo "    python scripts/run_msfold_sample.py \\"
echo "        --fasta examples/example.fasta \\"
echo "        --config configs/default.yaml \\"
echo "        --out outputs/test_full \\"
echo "        --device cuda --seed 42"
echo ""

echo "============================================"
echo "Smoke test PASSED"
echo "============================================"
