#!/usr/bin/env bash
set -euo pipefail

# setup_env.sh
# Creates a conda environment for the medical-segmentation-uncertainty project.
#
# Usage:
#   bash setup_env.sh              # auto-detect GPU if nvidia-smi exists
#   bash setup_env.sh gpu          # force CUDA PyTorch install
#   bash setup_env.sh cpu          # force CPU PyTorch install
#
# Optional environment variables:
#   ENV_NAME=medseg-uncertainty bash setup_env.sh
#   PYTHON_VERSION=3.11 bash setup_env.sh
#   CUDA_INDEX=cu128 bash setup_env.sh gpu

ENV_NAME="${ENV_NAME:-medseg-uncertainty}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_INDEX="${CUDA_INDEX:-cu128}"
MODE="${1:-auto}"

echo "Environment name: ${ENV_NAME}"
echo "Python version:   ${PYTHON_VERSION}"
echo "Install mode:     ${MODE}"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is not available in PATH."
    echo "Install Miniconda or Anaconda first, then rerun this script."
    exit 1
fi

# Make conda activation work inside a bash script.
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda environment '${ENV_NAME}' already exists."
else
    echo "Creating conda environment '${ENV_NAME}'..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

if [[ "${MODE}" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        MODE="gpu"
    else
        MODE="cpu"
    fi
fi

if [[ "${MODE}" == "gpu" ]]; then
    echo "Installing PyTorch with CUDA index: ${CUDA_INDEX}"
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_INDEX}"
elif [[ "${MODE}" == "cpu" ]]; then
    echo "Installing CPU-compatible PyTorch..."
    pip install torch torchvision torchaudio
else
    echo "Error: unknown mode '${MODE}'. Use one of: auto, gpu, cpu."
    exit 1
fi

echo "Installing MONAI and project dependencies..."
pip install \
    monai \
    nibabel \
    SimpleITK \
    scikit-image \
    scipy \
    scikit-learn \
    pandas \
    matplotlib \
    tqdm \
    pyyaml \
    tensorboard \
    ipykernel \
    jupyter

echo "Registering Jupyter kernel..."
python -m ipykernel install \
    --user \
    --name "${ENV_NAME}" \
    --display-name "Python (${ENV_NAME})"

echo "Verifying installation..."
python - <<'PY'
import torch
import monai
import nibabel
import SimpleITK as sitk

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch CUDA version:", torch.version.cuda)

print("MONAI:", monai.__version__)
print("Nibabel:", nibabel.__version__)
print("SimpleITK:", sitk.Version())
PY

echo "Writing reproducibility files..."
conda env export --from-history > environment.yml
pip freeze > requirements.txt

echo ""
echo "Done."
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "If CUDA available is False on a GPU machine, fix your NVIDIA driver before training."
