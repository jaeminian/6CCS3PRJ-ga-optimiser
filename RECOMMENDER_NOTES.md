# Recommender Notes (Optional) — OMSCS Letter Guidance

Thank you again for supporting my Georgia Tech OMSCS application. Georgia Tech notes that the most helpful recommendation letters focus on **technical Computer Science abilities** with **specific examples** (problem → approach/skills → outcome), rather than general personal characteristics.

This file is optional and is only here to make writing the letter easier. Please use (or ignore) anything you find useful.

---

## Project context (KCL)
- Module: **6CCS3PRJ Individual Project** (Final Year, King’s College London)
- Deliverables: working codebase + **55-page dissertation**

## One-paragraph project summary
My project implemented an end-to-end **Genetic Algorithm (GA)** system to optimise parameters for a rule-based trading strategy evaluated via historical backtesting. I built the full pipeline: market data ingestion and OHLC normalisation, technical-indicator signal generation (**RSI**, **candlestick pattern triggers**, **Williams %R**), a backtesting loop, and a GA optimiser (selection/crossover/mutation/elitism). Candidate strategies were scored using a **risk-adjusted objective** (Sharpe ratio derived from backtest statistics). I also parallelised fitness evaluation using Python multiprocessing to make the evolutionary search computationally tractable.

> Note: “predicting” refers to **parameter optimisation via backtesting**, not a standalone price-forecasting model.

## Technical points you may mention (pick any)
- End-to-end GA optimiser implementation (selection/crossover/mutation/elitism/generation loop)
- Configurable numeric gene vectors and parameter ranges
- Two-tier mutation (coarse resampling + bounded fine-tuning)
- Parallel fitness evaluation with multiprocessing (e.g., ProcessPoolExecutor)
- Risk-adjusted fitness objective (Sharpe ratio)
- Strategy signals: RSI, candlestick triggers, Williams %R
- Full OHLC data pipeline and reproducible experimentation

## Sample sentences (copy/paste)
- “Jaemin independently designed and implemented an end-to-end genetic algorithm pipeline for parameter optimisation, including selection, crossover, mutation, and elitism.”
- “He parallelised backtest-based fitness evaluations with multiprocessing, making the evolutionary search computationally tractable.”
- “He optimised a risk-adjusted objective using Sharpe ratio derived from full backtest statistics.”
- “He completed a substantial dissertation alongside a working implementation, demonstrating sustained independent technical work.”

## OMSCS alignment (closing options)
- “This project demonstrates readiness for graduate study through algorithm design, systems skills (parallelism), and rigorous empirical validation.”
