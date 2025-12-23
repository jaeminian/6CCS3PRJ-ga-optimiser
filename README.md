````markdown
# 6CCS3PRJ Individual Project (KCL) — Genetic Algorithm Strategy Parameter Optimiser

This repository contains my final-year Individual Project (6CCS3PRJ, King’s College London).  
I built an end-to-end Genetic Algorithm (GA) system to optimise parameters for a rule-based trading strategy, evaluated via historical backtesting using a risk-adjusted objective (Sharpe ratio).

## Quick links (for supervisors / recommenders)
- **1-page technical summary (PDF):** `docs/technical_summary_6CCS3PRJ.pdf`
- **Dissertation (PDF):** `docs/dissertation.pdf` *(or provide an external link if not stored in this repo)*
- **Optional recommender notes / sample phrasing:** `RECOMMENDER_NOTES.md`
- **Data notes (datasets not included):** `data/README.md`

> **Note:** In this project, “predicting” refers to **parameter optimisation via backtesting**, not a standalone price-forecasting model.

---

## Project Setup Instructions

This section outlines the steps required to set up the environment and run the project.

### Environment Setup
1. Create a new Conda environment:
   ```bash
   conda create --name myenv python=3.9
````

2. Activate the created environment:

   ```bash
   conda activate myenv
   ```
3. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

### Configuration and Operation

#### Data Preparation

1. Run the dataset preparation script:

   ```bash
   python src/generate_dataset.py
   ```

   You may customise the script if additional indicators or data fields are required.

#### Strategy Modification

1. Modify the trading strategy in:

   * `src/strategy_operation.py` (under the `Strategy` subclass)
2. Adjust the fitness function inside the backtesting routine in:

   * `src/strategy_operation.py`

#### Genetic Algorithm Optimisation

1. Configure the inputs in `JMstrategy` in:

   * `src/genetic_optimizer.py`
2. Run the optimiser:

   ```bash
   python src/genetic_optimizer.py
   ```

```
```
