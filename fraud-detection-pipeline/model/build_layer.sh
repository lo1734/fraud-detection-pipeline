#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Build script for Fraud Model AWS Lambda Layer
# Packages ML dependencies and model artifacts into fraud-model-layer.zip
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYER_DIR="${SCRIPT_DIR}/layer"
ARTIFACTS_DIR="${SCRIPT_DIR}/artifacts"

echo "=== Building Fraud Model Lambda Layer ==="
echo "Script directory:    ${SCRIPT_DIR}"
echo "Layer directory:     ${LAYER_DIR}"
echo "Artifacts directory: ${ARTIFACTS_DIR}"

# 1. Clean and create the layer directory structure
echo "--> Cleaning and creating layer directory structure..."
rm -rf "${LAYER_DIR}"
mkdir -p "${LAYER_DIR}/python/lib/python3.11/site-packages"
mkdir -p "${LAYER_DIR}/ml/model"

# 2. Install runtime dependencies into the python site-packages dir using pip
echo "--> Installing lightweight runtime dependencies (xgboost + numpy)..."
pip install \
    --target "${LAYER_DIR}/python/lib/python3.11/site-packages/" \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --python-version 3.11 \
    --only-binary=:all: \
    --no-deps \
    xgboost numpy

# Strip tests and __pycache__ to keep upload fast
find "${LAYER_DIR}/python" -type d -name "tests" -prune -exec rm -rf {} + 2>/dev/null || true
find "${LAYER_DIR}/python" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "${LAYER_DIR}/python" -name "*.dist-info" -prune -exec rm -rf {} + 2>/dev/null || true

# 3. Copy model artifacts from artifacts/ to layer/ml/model/
echo "--> Copying model artifacts..."
if [[ ! -f "${ARTIFACTS_DIR}/feature_config.json" ]]; then
    echo "Error: Required feature_config.json not found in ${ARTIFACTS_DIR}" >&2
    exit 1
fi

if [[ -f "${ARTIFACTS_DIR}/fraud_model.json" ]]; then
    cp "${ARTIFACTS_DIR}/fraud_model.json" "${LAYER_DIR}/ml/model/"
fi
if [[ -f "${ARTIFACTS_DIR}/fraud_model.joblib" ]]; then
    cp "${ARTIFACTS_DIR}/fraud_model.joblib" "${LAYER_DIR}/ml/model/"
fi
cp "${ARTIFACTS_DIR}/feature_config.json" "${LAYER_DIR}/ml/model/"

# 4. cd into layer/ and zip everything into ../fraud-model-layer.zip
echo "--> Packaging layer zip archive..."
rm -f "${SCRIPT_DIR}/fraud-model-layer.zip"
cd "${LAYER_DIR}"
zip -r ../fraud-model-layer.zip .
cd "${SCRIPT_DIR}"

# 5. Print the zip file size and success message
ZIP_PATH="${SCRIPT_DIR}/fraud-model-layer.zip"
ZIP_SIZE=$(du -h "${ZIP_PATH}" | cut -f1)

LAYER_SIZE=$(du -sh "${LAYER_DIR}" | cut -f1)

echo "=================================================================="
echo " Lambda layer built successfully!"
echo " Layer directory: ${LAYER_DIR}  (${LAYER_SIZE} — used by SAM)"
echo " Zip archive:    ${ZIP_PATH}  (${ZIP_SIZE} — for manual deploy)"
echo "=================================================================="
echo ""
echo " NOTE: The layer/ directory is kept for 'sam build' (ContentUri)."
echo "       The .zip is for manual 'aws lambda publish-layer-version'."
echo ""
echo "Done."
