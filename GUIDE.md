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

Local Git configuration to remove `.ipynb` outputs from commits
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

Run [prepare-data.ipynb](/prepare-data.ipynb) notebook to generate dataset

## 3. Train Model

Run [notebook.ipynb](/notebook.ipynb) notebook