# Project structure and reproducibility

```text
.
├── data/                 # Put ETTh1.csv here (not tracked by Git)
├── docs/                 # Project notes
├── notebooks/            # Original experiment notebook
├── presentation/         # Course presentation
├── reports/              # Project report and paper
├── results/              # Recorded metrics
├── src/
│   ├── data.py           # Chronological split, scaling, windows
│   ├── models.py         # LSTM, BiLSTM-attention, Transformer
│   └── train.py          # Training and test evaluation
├── .gitignore
├── README.md
└── requirements.txt
```

## Run a model
From the repository root:

```bash
pip install -r requirements.txt
python src/train.py --data data/ETTh1.csv --model lstm
python src/train.py --data data/ETTh1.csv --model attention
python src/train.py --data data/ETTh1.csv --model transformer
```

The recorded metric table in `results/recorded_model_metrics.csv` is from the supplied project materials. Running the scripts above creates a new experiment; do not assume it will reproduce those exact numbers. The notebook and report describe the supplied original experiment.
