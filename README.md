# pvf: PV AC-power forecasting research framework (SET Lab, MIST)

Build, change, fuse and compare deep forecasting models for PV AC power, using data from NWPGCL Sirajganj (SSP, 2025)
and the NIST ground array (2017). The candidate models come from `Resources/SET_Lab_Dl_Model_Selection.pdf`.
**Start with `notebooks/run_experiment.ipynb` for interactive experiments**, or use `tools/run_all_models.py` for a batch.

For setup from scratch, a file-by-file project map, and step-by-step experiment instructions, see the
[setup and experiment guide](docs/experiment_guide.md).

## Quick start

Use Python 3.11 and VS Code with Microsoft's Python and Jupyter extensions. Install [Git](https://git-scm.com/install/windows), then clone:

```powershell
git clone https://github.com/TanvirSahed/Solar_pvp_forecasting.git
cd Solar_pvp_forecasting
```

Obtain the processed `Data/` folder from the project owner and copy it into this folder before setup.
Datasets are shared separately; GitHub does not include them. The [guide lists the four required Parquet files](docs/experiment_guide.md#16-obtain-the-datasets-separately).
Pretrained model weights download automatically when first needed.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1
```

On Linux, run `bash setup_venv.sh` instead. Setup selects a GPU or CPU build automatically.
Open this folder in VS Code, open `notebooks/run_experiment.ipynb`, select the `.venv` kernel,
then follow the **CHOOSE** cells from top to bottom. No Docker installation is needed.

## How the pieces connect

```
 data pipeline              input adapter               model architecture          training recipe
 pvf/data/                  pvf/model_inputs/           pvf/models/                 pvf/training/
 DataConfig ──► ForecastBatch ──► model input ──► forecast ◄── loss, optimizer, scheduler
 (site, features,           (named steps:               (hyperparameters =          (TrainingRecipe)
  lookback, split)           trace() / plot())           constructor arguments)
                                                                   │
                               pvf/experiment.py  run_experiment() ┘──► runs/<site>/<run>/<time>/  (model + results)
                               pvf/evaluation/    metrics, compare_runs(), diagnostic plots
```

| Module | Holds | Change it to... |
|---|---|---|
| `pvf/data/features.py` | named features (`FEATURES`) | add a feature: write one function and list it |
| `pvf/data/pipeline.py` | `DataConfig` -> `build_data_bundle()` -> train/validation/test datasets + `DataSpec` | choose site, features, lookback, horizon |
| `pvf/data/splits.py` | `SplitConfig`: blocked (every season in every split) or by month, plus holdout | change the evaluation protocol |
| `pvf/model_inputs/` | `InputAdapter` (a list of named steps) and the steps themselves | reshape the data for a new architecture |
| `pvf/models/architectures/` | one family per file, every model registered by name | change hyperparameters, layers, connections |
| `pvf/models/blocks/` | shared building blocks: Mamba, graph layers, heads, RevIN, KAN... | reuse parts across models |
| `pvf/models/base.py` | `ForecastModel`, the contract between models and training | (rarely) |
| `pvf/training/` | losses, optimizers, schedulers (by name), `TrainingRecipe`, trainer, `paper_recipes.py` | add a loss, optimizer or scheduler; change a paper preset |
| `pvf/evaluation/` | standard metrics, conformal intervals, run comparison and plots | add a metric or diagnostic plot |
| `pvf/experiment.py` | `run_experiment`, `load_trained_model` | (rarely) |
| `pvf/search.py` | `search_hyperparameters` (small grid / random search, ranked on validation) | (rarely) |
| `pvf/workspace.py` | `check_environment` | (rarely) |

### The model contract (`pvf/models/base.py`)
Every model is called directly on a data batch, `model(forecast_batch)`. Inside, it runs
`self.input_adapter` and then `self.forecast(model_input)`, which returns `{"forecast": ..., "auxiliary_loss"?: ...}`.
Models are written as `encode()` (the reusable part) followed by a forecast head. The trainer touches nothing else,
so any architecture that follows this contract trains, saves and is scored the same way.

### Every experiment is saved
`runs/<site>/<run_name>/<timestamp>/` contains:
- `trained_model.pt`: the whole model, reloaded with `pvf.load_trained_model(run_dir)`;
- `best_model.pt`: the best weights;
- `config.yaml`: the data config, model, hyperparameters and training recipe the run used;
- `metrics.json`: training history (with validation nMAE/nRMSE per epoch) and overall and per‑horizon scores per split;
- `predictions_<split>.npz`.

Each run also adds one row to `results/experiment_log.csv`.

### Training controls (`pvf.TrainingRecipe`)
Loss (+ options), optimizer (+ options), learning rate, weight decay, scheduler (+ options), warm‑up epochs, epochs,
early‑stopping metric (validation loss, nMAE or nRMSE) and patience (`None` = fixed epochs), batch size, gradient
accumulation, gradient clipping, mixed precision, training‑data fraction, initialising from another run, and
freezing parameter groups.

**Paper recipes:** `pvf.paper_recipe("<model>")` loads the hyperparameters, training recipe and lookback each
reference paper used. Values a paper leaves out are filled in and listed in `.notes`. The full write‑up is in
`docs/training_recipes.md`, with a PDF copy in `Resources/Training_Recipes_Reference_Papers.pdf`; regenerate it
with `python tools/markdown_to_pdf.py docs/training_recipes.md Resources/Training_Recipes_Reference_Papers.pdf`.

**Train every model at once:** `python tools/run_all_models.py --site ssp --time-limit 15` trains each model with its
paper recipe (at most 15 min each), scores all of them on the same test windows, and writes
`Resources/Experiment_Report_<SITE>_<date>.md` and `.pdf`: a ranked summary table, then per model its features,
recipe, epochs and loss graph. Use `--report-only` to rebuild the report from the last batch.

### Standard scores (identical for every model)
All scores are on daytime targets, as a percentage of rated power:
- nMAE, nRMSE, bias, 95th‑percentile error, R²;
- skill versus smart persistence (> 0 means the model beats the physics reference);
- ramp error;
- the same metrics per forecast step;
- for quantile models, also CRPS, pinball loss and interval coverage (PICP) and width (PINAW).

Definitions are in `pvf.evaluation.metrics.METRIC_DESCRIPTIONS`.

## Naming rules
PEP 8, so every kind of name is recognisable at a glance and matches PyTorch, numpy and pandas:

| Kind | Case | Example |
|---|---|---|
| class | PascalCase | `GraphMamba`, `TrainingRecipe`, `BuildVariableNodes` |
| function, method, variable, argument | snake_case | `build_data_bundle`, `history_features`, `lookback_steps` |
| constant | UPPER_SNAKE | `FEATURES`, `MODEL_REGISTRY`, `CALENDAR_FEATURE_NAMES` |
| registry name (model, loss, ...) | snake_case string | `"hdcf_mamba"`, `"reduce_on_plateau"` |

Use full words, not `x`, `h` or `cfg`. Put units in the name (`ac_power_kw`, `irradiance_wm2`). Add tensor
shapes as comments (`# [batch, lookback, n_features]`).

## Adding things

**A model:** add a class to a file in `pvf/models/architectures/`:

```python
@register_model("my_model")
class MyModel(ForecastModel):
    def __init__(self, data_spec, model_size=64, dropout=0.1, input_adapter=None):  # hyperparameters
        super().__init__(data_spec, input_adapter)          # or pass an InputAdapter([...steps]) as the default
        self.encoder = ...
        self.forecast_head = ForecastHead(model_size, data_spec.horizon_steps, data_spec.n_future_features)
    def encode(self, model_input):
        return self.encoder(model_input["history_features"])           # [batch, model_size]
    def forecast(self, model_input):
        return {"forecast": self.shape_output(self.forecast_head(self.encode(model_input),
                                                                 model_input["future_features"]))}
```

New files must be imported in `pvf/models/architectures/__init__.py`. `python -m pytest -q tests` then checks that every model trains one step.

**Fusing two models:** follow `pvf/models/architectures/fusion_example.py`, which fuses iTransformer's variable
encoder with HDCF‑Mamba's temporal encoder through a learned gate. Its docstring gives the 4‑step recipe; copy it
and swap in the two encoders you want.

**A feature:** one `Feature(...)` entry in `pvf/data/features.py`. Mark `known_in_advance=True` only if its
future values really are known at forecast time.

**An input step:** subclass `AdapterStep` in `pvf/model_inputs/adapter_steps.py`. It adds named tensors to the
input and keeps the rest. Steps adapt to the selected features and skip missing ones with a note.

**A loss, optimizer or scheduler:** one line in `pvf/training/losses.py` or `pvf/training/optimizers.py`.

## Environment notes
Needs Git and Python 3.11. An NVIDIA GPU and driver are optional; the same notebook also runs on CPU.

The script installs the exact package versions from `requirements/lock-*.txt`. It picks the PyTorch CUDA build
from the driver: CUDA 13.0 for driver 580 or newer, CUDA 12.6 for older drivers, otherwise CPU. It then runs
`pvf.check_environment()`, which prints READY when packages and data files are present on either CPU or GPU.
To explicitly install the CPU build, use `powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1 -Cuda cpu`
(Linux: `bash setup_venv.sh cpu`). Open `notebooks/run_experiment.ipynb` with the `.venv` kernel.
Training selects CUDA when available, otherwise CPU with full precision. GPUs without bf16 support use fp16 automatically.
On Windows, recurrent layers use PyTorch's native implementation on the selected device to avoid a shutdown crash
in the pinned cuDNN runtime; GPU training remains available but may be slower for LSTM/GRU.

**Sharing experiments:** commit code changes and push; your colleague pulls and runs the notebook. A run folder can
be copied over and loaded with `pvf.load_trained_model(run_dir)`.

## Data notes
- NIST timestamps follow local time **with** daylight saving (America/New_York). The mean irradiance peak is at
  about 12:05 in January and about 12:54 in June, so a fixed GMT−5 would shift clear sky by 1 h from March to November.
- Each site has one inverter, while the selection report's graph model assumes four. For now `graph_mamba` uses
  the variables as nodes. Write a `BuildInverterNodes` adapter step once per‑inverter data arrive.
- There is no weather‑forecast (NWP) archive yet. The `*_oracle` future features are perfect‑forecast upper bounds,
  not real forecasts.
- Smart persistence is weak at long horizons from night origins (it predicts 0), which inflates skill scores at
  day‑ahead horizons. Read skill together with nMAE and nRMSE.

Paper links are in `ref/README.md`; the papers themselves travel with the data.
