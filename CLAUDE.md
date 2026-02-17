# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚡️ Initialization (The Standard Protocol)

**Before starting work:**

1. **Activate Venv:** `source /home/griermarkov/Drives/.virtual-environments/MUPLab_env/bin/activate`
2. **Initialize:** Run `/init` to sync and update dependencies.

### /init Command

```bash
# Pull latest changes and install dependencies
git pull origin $(git branch --show-current) && \
  pip install -r requirements.txt 2>/dev/null || \
  pip install -e . 2>/dev/null || \
  echo "No installable dependencies found"
```

### /save-point Command

Smart commit workflow that respects .gitignore and pushes to remote:

```bash
# Phase XV: Federation Save Point
git status
git add .
git commit -m "Phase XV: Federation Infrastructure Update"
git push origin HEAD
```

**Usage:** Run `/save-point` to checkpoint your work.


---

## Project: MUPLab
> **Profile:** rjwingate
> **Identity:** rjwingate / id_ed25519_rjwingate
> **Stack:** Python, Jupyter Notebooks, NumPy, SciPy, scikit-learn, librosa, soundfile, matplotlib

## Description
Moringa Ultrasonic Pop Classification research using 384 kHz ultrasound spectral analysis with RMS thresholding and ML classifiers on Raspberry Pi 5.

## R&D Team Assignments
* **Architect:** `system-architect` (Use for pipeline architecture)
* **Builder:** `polyglot-dev` (Use for Python/ML implementation)
* **Researcher:** `quantum-ag-researcher` (Use for bioacoustics research and ML theory)
* **Ops:** `infrastructure-ops` (Use for Pi deployment)
* **Guardian:** `security-qa` (Use for testing)

## Commands
* **Build:** N/A (Jupyter notebook-based)
* **Test:** `jupyter nbconvert --to notebook --execute notebooks/MVPOC_ultrasonic_pop_classifier.ipynb`
* **Lint:** N/A

## Hardware Stack
- Raspberry Pi 5 (16GB)
- Dodotronic Ultramic 384K EVO
- Samsung 990 PRO 1TB NVMe
- BME280/TSL2591/SGP40 sensors
- VIVOSUN VGrow Smart Box

## Federation Context
* **Root:** `~/Drives/data/SynologyDrive/grimmerie/repositoriies-by-profile`
* **Sibling Projects:** You are authorized to traverse `../` to reference other repos in the rjwingate profile.
* **Related Repos:** easaixdvil/MUPLab (production ML pipeline), easaixdvil/PlantWaveMonitor (bio-sensing)
