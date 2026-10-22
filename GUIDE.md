# Guide

This is a quick guide to get started on the project.

## Setup Environment

Requires Python 3.10+

Create virtual environment and activate it
```bash
# Linux/MacOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

Install requirements
```bash
pip install -r requirements.txt
```

For development, setup the local Git configuration to remove `.ipynb` notebook outputs from commits
```bash
nbstripout --install
```

## Prepare Dataset

Download Cityscapes datasets: https://www.cityscapes-dataset.com/downloads.
- gtFine_trainvaltest.zip (241MB)
- leftmg8bit_trainvaltest.zip (11GB)

Unzip and place inside `data/`

data/   
├──gtFine_trainvaltest/   
└──leftImg8bit_trainvaltest/    

Run [1-dataset-creation.ipynb](/notebooks/1-dataset-creation.ipynb) to generate dataset

Run [2-exploratory-data-analysis.ipynb](/notebooks/2-exploratory-data-analysis.ipynb) to analyse dataset

## Train Model

Run [3-model-training.ipynb](/notebooks/3-model-training.ipynb) to train the model


## Troubleshooting CUDA

If you own a supported NVIDIA GPU, it is recommented to use CUDA. PyTorch should show device `cuda` instead of device `cpu`. It can accelerate model training and inference significantly.

Make sure you have installed CUDA from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads). Run `nvidia-smi` in terminal and check top-right corner for "CUDA Version". This is the maximum CUDA Driver version that your installed NVIDIA graphics driver supports.

If you installed PyTorch through the requirements, it often fetches the cpu-only verison. Reinstall it:

```
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```