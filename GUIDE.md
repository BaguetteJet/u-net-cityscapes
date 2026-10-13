# Guide

## 1. Setup Environment

Create virtual environment and activate
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

Local Git configuration to remove `.ipynb` notebook outputs from commits
```bash
nbstripout --install
```

## 2. Prepare Data

Download Cityscapes datasets: https://www.cityscapes-dataset.com/downloads
- gtFine_trainvaltest.zip (241MB)
- leftmg8bit_trainvaltest.zip (11GB)

Unzip both into data/:

data/   
├──gtFine_trainvaltest/   
└──leftImg8bit_trainvaltest/    

Run [1-dataset-creation.ipynb](/notebooks/1-dataset-creation.ipynb) notebook to generate dataset

Optionally run [2-exploratory-data-analysis.ipynb](/notebooks/2-exploratory-data-analysis.ipynb) to analyse dataset

## 3. Train Model

Run [3-model-training.ipynb](/notebooks/3-model-training.ipynb) notebook