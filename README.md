# Recipe LSTM: Autoregressive Text Generation (Foster Ch.5, Option A)

Word-level LSTM text generator on the epirecipes corpus. The model, sampling
routine, and data flattening follow David Foster, *Generative Deep Learning*
(2nd ed.), O'Reilly, 2023, Chapter 5, notebook `05_autoregressive/01_lstm`.

## What is here

| Path | What it is |
|---|---|
| `lstm_recipes.py` | Data loading, vectorisation, model, sampling, analysis helpers |
| `run_train.py` | Full training run and artefact generation (~27 min on CPU) |
| `lstm_recipes_assignment.ipynb` | Submitted notebook, all cells run with figures |
| `WRITEUP.pdf` | 1-2 page write-up |
| `scripts/build_notebook.py` | Regenerates the notebook from plain-text cell sources |
| `scripts/build_report.py` | Regenerates the PDF from the run artefacts |
| `data/full_format_recipes.json` | epirecipes dataset, 20,130 records |
| `artifacts/` | Loss history, curve, samples, metrics, training log |
| `models/lstm.keras`, `models/vocab.json` | Trained weights and vocabulary |

## Reproduce

```bash
# 1. Dataset (Kaggle, anonymous download, no credentials needed)
curl -L -o epi.zip https://www.kaggle.com/api/v1/datasets/download/hugodarwood/epirecipes
unzip -o epi.zip full_format_recipes.json -d data/

# 2. Environment (reuses the Module 2/3 venv: TensorFlow 2.21, Keras 3, Python 3.13)
../vae-fashion/.venv/bin/python -m pip install -r requirements.txt

# 3. Train and write artefacts (~27 min on Apple-silicon CPU)
../vae-fashion/.venv/bin/python run_train.py

# 4. Build the notebook and the PDF
../vae-fashion/.venv/bin/python scripts/build_notebook.py
../vae-fashion/.venv/bin/python scripts/build_report.py
```

## Run configuration

| Setting | Value |
|---|---:|
| Corpus | 20,098 usable recipes, `Recipe for {title} | {directions}` |
| Vocabulary | 10,000 word tokens, lowercased |
| Max length | 200 tokens (+1 for the shifted target) |
| Embedding dim | 100 |
| LSTM units | 128, `return_sequences=True` |
| Output | `Dense(10000, softmax)` |
| Parameters | 2,407,248 |
| Loss | sparse categorical cross-entropy |
| Optimiser | Adam, default learning rate |
| Batch size | 32 |
| Epochs | 12 (Foster's notebook uses 25; loss is near-flat well before then) |
| Seed | 42 |
| Hardware | Apple silicon CPU, no Metal |

Weights are saved at the end of every epoch to `checkpoint/lstm.keras`, so a run
stopped early still leaves a usable model.

## Outputs

- `artifacts/history.csv` — epoch, loss, wall-clock seconds
- `artifacts/loss_curve.png` — training loss against the `log(VOCAB_SIZE)` uniform floor
- `artifacts/samples.json` — full passages at T = 0.5, 1.0, 1.5
- `artifacts/temperature_report.csv` — confidence and diversity statistics per temperature
- `artifacts/metrics.json` — environment, parameters, timings, final loss
- `artifacts/train.log` — full training log including one sample per epoch

## The finding

Training loss falls from about 9.2 (near the uniform-guess floor) and continues
to descend through the final epoch, so the model stays underfit rather than
overfitting. Sampling the same prompt at three temperatures shows the
temperature-quality tradeoff directly: mean probability of the chosen token
falls monotonically with temperature while the unique-token ratio rises, and
coherence breaks at clause boundaries first at T = 1.5. Numbers and quoted
passages are in `WRITEUP.pdf`.
