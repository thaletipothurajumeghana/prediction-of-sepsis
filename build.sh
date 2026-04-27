#!/usr/bin/env bash
# Render build script
set -e

echo "=== SepsisAI Build ==="
echo "Python: $(python --version)"
echo "Pip:    $(pip --version)"

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify model files exist
echo ""
echo "=== Checking model artifacts ==="
for f in best_model.pkl scaler.pkl imputer.pkl metadata.json; do
  if [ -f "backend/models_store/$f" ]; then
    echo "  OK  $f"
  else
    echo "  MISSING: $f — model must be committed to the repo"
    exit 1
  fi
done

echo ""
echo "=== Build complete ==="