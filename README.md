# Enhanced LSTM-Based Time Series Forecasting of Transformer Oil Temperature

**Student:** Kadari Anirudh  
**Hall Ticket:** 2403A52193  
**Institution:** SR University  
**Academic Year:** 2026–2027

## Overview
This project forecasts transformer oil temperature (ETTh1 target column `OT`) from seven time-series variables. It compares a baseline LSTM, a bidirectional LSTM with additive attention, and a Transformer encoder.

## Forecast setup
- Historical input window: 96 hourly steps
- Forecast horizon: 24 hourly steps
- Features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
- Chronological split: 70% train, 10% validation, 20% test
- Scaling: StandardScaler fitted on training data only

## Recorded evaluation results
| Model | MSE | MAE | RMSE |
|---|---:|---:|---:|
| Baseline LSTM | 3.4520 | 1.4252 | 1.8580 |
| Improved LSTM + Attention | 3.7545 | 1.4979 | 1.9376 |
| Transformer | 3.7411 | 1.5190 | 1.9342 |

These are the recorded results supplied with the project; they have not been independently rerun during repository publication.

## Repository contents
- `notebooks/`: original experiment notebook (to be added from the supplied project archive)
- `src/`: reusable model definitions and training utilities
- `data/`: place the ETTh1 CSV here; large raw datasets are not committed
- `reports/`: project report and research paper
- `presentation/`: project presentation

## Dataset
Use the ETTh1 dataset from the official ETDataset repository: https://github.com/zhouhaoyi/ETDataset  
Place `ETTh1.csv` at `data/ETTh1.csv`.

## Run
```bash
pip install -r requirements.txt
python src/train.py --data data/ETTh1.csv
```
