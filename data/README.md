# Data Requirements

This project utilizes the **NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) Turbofan Engine Degradation Dataset** (specifically `FD001`).

### How to acquire the data:
1. Download the dataset from NASA's official open-data repositories or Kaggle.
2. Extract and place `train_FD001.txt` and `test_FD001.txt` directly into this `data/` folder.

The data layout expects space-separated files containing 26 columns: engine ID, operational cycles, 3 operational settings, and 21 distinct sensor monitoring readouts.
