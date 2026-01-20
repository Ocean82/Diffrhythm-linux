#!/bin/bash
echo "Installing GPU-enabled PyTorch..."
cd /mnt/d/EMBERS-BANK/DiffRhythm-LINUX
source .venv/bin/activate

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Testing CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"