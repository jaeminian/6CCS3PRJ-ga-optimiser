
# Project Setup Instructions

This document outlines the steps required to set up the environment and run the project successfully. Follow these steps in the order provided.

## Environment Setup


// 시작하는 방법
// cd 로 들어가준다.

1. Create a new Conda environment:
   ```bash
   conda create --name myenv python=3.9
   ```
2. Activate the created environment:
   ```bash
   conda activate myenv
   ```
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration and Operation

### Data Preparation

1. Run `generate_dataset.py` to collect and prepare the data. Customize the script if certain indicators are missing.

### Strategy Modification

1. Modify the trading strategy in `strategy_operation.py` under the `Strategy` subclass.
2. Define your own fitness function within the backtesting function in `strategy_operation.py`.

### Genetic Algorithm Optimization

1. Customize the inputs in your `JMstrategy` class in `genetic_optimizer.py` based on the number

//python genetic_optimizer.py