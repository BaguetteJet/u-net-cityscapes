# Guide

## Setup

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