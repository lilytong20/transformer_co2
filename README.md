# Transformer-based CO2 Profile Estimation

Transformer model for forecasting CO2 concentrations at 6 absorber sampling points in a carbon capture pilot plant.

## Setup

```bash
conda env create -f environment.yml
conda activate co2_transformer
```

Or: `pip install -r requirements.txt`

## Training & Analysis

Open `notebooks/train_and_evaluate_colab.ipynb` in Colab or Jupyter:
- Single training run and full 3x4 benchmark grid (input windows {15, 17, 19} x forecast windows {1, 4, 8, 18})
- Results saved to `results/benchmark_results.csv`

Open `notebooks/analysis_colab.ipynb` (after training):
- Attention maps, gradient-based feature importance, error analysis

## Model

Encoder-only Transformer, all components implemented from scratch:

```
Input (batch, 19, 96) — 90 sensors + 6 one-hot label
  -> Linear(96, 64) -> Sinusoidal PE -> 2x TransformerBlock (pre-norm, 4 heads)
  -> LayerNorm -> Temporal Attention Pooling
  -> Linear(64, 128) -> GELU -> Linear(128, fw*6) -> Sigmoid
Output (batch, fw, 6) — CO2 at 6 sampling points
```

Best config: input_window=19, forecast_window=18, RMSE = 0.1619 CO2 %.

## Deployment

Local test (no Docker, no DB):

```bash
cd transformer_co2
python deployment/test_api.py
```

Or start a server:

```bash
uvicorn deployment.app:app --reload --port 8000
python deployment/test_api.py --http   # in another terminal
```

With Docker + PostgreSQL:

```bash
cd transformer_co2/deployment
docker compose up --build
```

## Data

8 experimental datasets, 90 process sensors each.
Test set: `140207_1.xlsx`, Training: remaining 7 files.
