# Legislative Bill Passage Prediction Model

Predict the probability that a U.S. state legislature bill will be enacted into law. Currently targeting Ohio (131st-136th General Assembly, 2015-2026), with a state-agnostic architecture designed to be replicated for any state available through LegiScan.

The model produces a **calibrated probability** (0-1) with **interpretable factor attribution** (SHAP values) — designed for policy professionals, lobbyists, and legislative analysts who need to understand *why* a bill is likely or unlikely to pass, not just the number.

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A [LegiScan API key](https://legiscan.com/legiscan) (free public tier: 30,000 queries/month)

### 2. Install

```bash
pip install -e ".[dev]"
```

### 3. Configure API Key

Create a `.env` file in the project root (see `.env.example`):

```
LEGISCAN_API_KEY=your_api_key_here
```

Never commit this file. It is already in `.gitignore`.

### 4. Load Historical Data

Download and normalize Ohio legislative data into the local SQLite database. This fetches bulk dataset archives from LegiScan for sessions 131-136 (2015-2026):

```bash
python -m src.cli load-data --state OH
```

To load specific sessions only:

```bash
python -m src.cli load-data --state OH --sessions 134,135,136
```

This creates `data/ohio_legislature.db` containing bills, sponsors, committees, history events, roll calls, and individual votes.

### 5. Train the Model

Train on historical sessions (default: 131st-134th GA):

```bash
python -m src.cli train --state OH
```

With custom session range or model type:

```bash
python -m src.cli train --state OH --sessions 131,132,133,134,135 --model-type xgboost
python -m src.cli train --state OH --model-type logistic
```

Filter to specific bill types (e.g., only House and Senate Bills):

```bash
python -m src.cli train --state OH --bill-types HB,SB
```

Model artifacts are saved to `models/ohio/`.

### 6. Evaluate

Evaluate against a held-out session using temporal validation (default: 135th GA):

```bash
python -m src.cli evaluate --state OH --test-session 135
```

This reports:
- **AUC-ROC** and **AUC-PR** (precision-recall, the primary metric given class imbalance)
- **Brier score** (calibration quality)
- **Log loss**
- **ECE** (Expected Calibration Error)

Evaluation plots (ROC curve, PR curve, calibration plot) are saved to `models/ohio/`.

### 7. Predict a Bill

By bill number:

```bash
python -m src.cli predict --state OH --bill "HB 123"
```

By LegiScan bill ID:

```bash
python -m src.cli predict --bill-id 1234567
```

Output includes:
- Calibrated probability of enactment
- Two-stage breakdown (P(exits committee) and P(enacted | exits committee))
- Comparison to historical base rate for this bill type
- Top factors helping and hurting the bill (SHAP attribution)
- Plain-English narrative summary with caveats

### 8. Inspect Features

View the computed feature vector for any bill:

```bash
python -m src.cli features --bill-id 1234567
python -m src.cli features --bill-id 1234567 --json-output
```

### 9. Sync Current Session

Incrementally update the current session data (only re-fetches bills whose `change_hash` has changed):

```bash
python -m src.cli sync --state OH --session 136
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `load-data` | Download and load historical data from LegiScan |
| `sync` | Incrementally sync the current session via change_hash |
| `train` | Train the passage prediction model |
| `evaluate` | Evaluate model on a held-out session |
| `predict` | Predict passage probability for a specific bill |
| `features` | Inspect computed features for a bill |

All commands support `--verbose` / `-v` for debug logging.

## How It Works

### Two-Stage Prediction Architecture

Following [GovTrack's methodology](https://www.govtrack.us/about/analysis), the model uses two stages because the dynamics that get a bill through committee are different from those that get it enacted:

1. **Stage 1**: P(bill exits committee) — trained on all bills
2. **Stage 2**: P(bill is enacted | exited committee) — trained only on bills that passed committee

**Combined**: P(enacted) = P(exits committee) x P(enacted | exits committee)

### 47 Engineered Features

| Category | Count | Key Features |
|----------|-------|-------------|
| **Sponsor** | 13 | Majority party, leadership position, seniority, historical success rate, bipartisan cosponsorship score, cosponsors on assigned committee |
| **Committee** | 6 | Historical pass-through rate, hearing count, chair alignment with sponsors |
| **Bill** | 20 | Progress stage, days since last action (staleness), roll call results, amendments, companion bill detection, text length, appropriations flag |
| **Session** | 11 | % session elapsed, election year, partisan composition, trifecta/supermajority, historical base rates by bill type and committee |

Features are derived from the political science literature on legislative prediction (GovTrack Prognosis, Nay 2016/Skopos Labs, Yano/Smith/Wilkerson 2012, VPF Framework 2025).

### Models

- **Primary**: XGBoost gradient boosted trees with `scale_pos_weight` for class imbalance
- **Baseline**: Logistic Regression with balanced class weights
- **Calibration**: Isotonic regression (default) or Platt scaling applied to raw model outputs
- **Interpretability**: SHAP (SHapley Additive exPlanations) for per-prediction feature attribution

### Evaluation Philosophy

- **Never accuracy alone** — with ~90%+ bills failing, always predicting "fail" gives trivially high accuracy
- **AUC-PR is the primary metric** — sensitive to performance on the minority (passed) class
- **Temporal validation only** — train on earlier sessions, test on later ones. No random cross-validation across time.
- **Calibration matters** — a model that says "30% chance" should see ~30% of those bills actually pass

## Project Structure

```
├── CLAUDE.md                     # Project context and methodology
├── .env                          # API key (not committed)
├── .env.example                  # Template for .env
├── pyproject.toml                # Dependencies and build config
├── data/
│   ├── raw/                      # Cached API responses and bulk datasets
│   ├── processed/                # Feature matrices
│   └── ohio_legislature.db       # SQLite database (created by load-data)
├── models/
│   └── ohio/                     # Trained model artifacts and eval plots
├── src/
│   ├── cli.py                    # Click CLI (train/evaluate/predict/load-data/sync)
│   ├── config.py                 # All configuration and hyperparameters
│   ├── predict.py                # Single-bill prediction with narrative output
│   ├── data/
│   │   ├── legiscan_client.py    # LegiScan API client (cache + rate limit)
│   │   ├── ohio_loader.py        # Bulk and per-bill data loading → SQLite
│   │   └── schema.py             # SQLAlchemy ORM schema
│   ├── features/
│   │   ├── build_features.py     # Feature matrix orchestrator
│   │   ├── sponsor_features.py   # Sponsor and cosponsorship features
│   │   ├── committee_features.py # Committee features
│   │   ├── bill_features.py      # Bill lifecycle and content features
│   │   └── session_features.py   # Session context and base rate features
│   └── models/
│       ├── passage_model.py      # Two-stage XGBoost/LogReg with calibration
│       └── evaluate.py           # Metrics, calibration, SHAP, plots
└── tests/
    ├── test_legiscan_client.py   # API client tests (mock responses)
    ├── test_features.py          # Feature computation tests (in-memory DB)
    └── test_model.py             # Model training/eval smoke tests (synthetic data)
```

## Database Schema

The SQLite database (`data/ohio_legislature.db`) contains:

| Table | Description |
|-------|-------------|
| `sessions` | Legislative sessions (e.g., 136th GA) |
| `bills` | Bill metadata, status, progress, text length |
| `legislators` | Legislator profiles (party, role, district) |
| `bill_sponsors` | Primary sponsors and cosponsors per bill |
| `bill_history` | Every status/action event with dates |
| `committees` | Committee names and chambers |
| `committee_referrals` | Bill-to-committee assignments |
| `roll_calls` | Aggregate vote tallies per roll call |
| `votes` | Individual legislator votes |
| `amendments` | Amendment records |

## Interpreting Predictions

When the model outputs a prediction, keep in mind:

1. **Base rate context is essential.** A 15% probability may sound low, but if the historical passage rate for this bill type is 4%, that bill is ~4x more likely to pass than average.

2. **Predictions are point-in-time.** Re-run as the bill accumulates new actions (committee hearings, amendments, floor votes). Use `sync` to update the data first.

3. **The model cannot predict political shocks** — sudden crises, scandal, federal preemption, court rulings, or a governor changing position.

4. **Ohio-specific dynamics apply.** The Republican supermajority means majority-party sponsorship and leadership alignment are particularly strong signals. Bipartisan cosponsorship may matter less for passage but signals broader support.

5. **"Incorporated" bills are a blind spot.** A bill may die formally but have its language folded into an omnibus or budget bill. The model predicts the bill-as-introduced passing, not its substantive content being enacted through another vehicle.

## Development

### Running Tests

```bash
python -m pytest tests/ -v
```

The test suite (49 tests) covers:
- API client: caching, error handling, change_hash sync (mock responses)
- Feature engineering: all 4 feature categories against an in-memory SQLite DB with known test data
- Model: training, prediction, save/load, calibration, evaluation (synthetic data)

### Configuration

All configurable parameters are in `src/config.py`:
- API settings (base URL, rate limits, cache directory)
- Database path
- Ohio session metadata and partisan composition
- Feature engineering constants
- Model hyperparameters (XGBoost, Logistic Regression)
- Calibration settings
- Train/validation/test session splits

### Adding a New State

The architecture is state-agnostic by design. To add a new state:

1. Add session metadata to `src/config.py` (session years, partisan composition)
2. Create a new loader modeled on `ohio_loader.py` (or generalize it)
3. Train a state-specific model: `python -m src.cli train --state XX`
4. Models are saved per-state under `models/{state}/`

## Roadmap

- **Phase 2**: NLP features from bill text (topic embeddings, subject classification)
- **Phase 3**: Multi-state support (parameterized loaders, per-state models)
- **Phase 4**: Live monitoring dashboard (scheduled re-scoring as bills advance)
- **Phase 5**: Individual legislator vote prediction
- **Phase 6**: OpenStates API integration for additional data enrichment
