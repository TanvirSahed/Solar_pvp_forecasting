# Setup and experiment guide

This guide covers, in order: setting up a new machine, understanding the files, and running experiments. Python and VS Code are assumed to be installed. The main instructions use **Windows PowerShell**; a Linux setup is included at the end.

Run terminal commands from the project root, the folder containing `pyproject.toml`, `setup_venv.ps1`, and `pvf/`. Run Python examples in a notebook cell unless a block is labelled `powershell` or `bash`.

## 1. Set up a new machine

### 1.1 Check the installed Python

Open **Terminal > New Terminal** in VS Code and select PowerShell. Check:

```powershell
py -0
py -3.11 --version
```

Use **Python 3.11** for the supplied setup scripts and pinned dependencies. Although `pyproject.toml` allows Python 3.12 too, the scripts and environment checker expect 3.11. If your installed Python is a different version, add 3.11 before continuing. If `python --version` works but `py` does not, the Windows setup script also needs the Python launcher.

### 1.2 Download Git

Download and install [Git for Windows](https://git-scm.com/install/windows), using the installer defaults. Alternatively, if Windows Package Manager is available:

```powershell
winget install --id Git.Git -e --source winget
```

Close and reopen the terminal, then check:

```powershell
git --version
```

### 1.3 Install the VS Code extensions

Open **Extensions** with `Ctrl+Shift+X`. Install these Microsoft extensions:

| Extension | Identifier | Purpose |
|---|---|---|
| [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) | `ms-python.python` | Select the project interpreter and edit Python |
| [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) | `ms-toolsai.jupyter` | Open and execute the experiment notebook |

You can also install them from a terminal if the `code` command is available:

```powershell
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
```

Notebook kernel selection is described in the [VS Code notebook documentation](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

### 1.4 Prepare the GPU, or choose CPU

An NVIDIA GPU is recommended for training. Download a driver for your GPU and Windows version from [NVIDIA's official driver page](https://www.nvidia.com/en-us/drivers/), install it, and restart if requested. Then run:

```powershell
nvidia-smi
```

This should show your GPU and driver version. The CUDA version shown here describes driver capability; it is separate from the CUDA runtime used by PyTorch.

The supplied setup script installs a prebuilt PyTorch package and selects its CUDA runtime from the driver. This project uses `mambapy` and does not require compiling a separate Mamba CUDA extension. A separate CUDA Toolkit installation is not part of the native setup below. Use the project's pinned requirements rather than replacing them with the latest command from a generic PyTorch tutorial.

If there is no NVIDIA GPU, the same setup command in section 1.7 selects CPU automatically. Start with a small LSTM experiment because larger runs can be slow.

### 1.5 Obtain the project

Clone the [project repository](https://github.com/TanvirSahed/Solar_pvp_forecasting):

```powershell
git clone https://github.com/TanvirSahed/Solar_pvp_forecasting.git
Set-Location -LiteralPath '.\Solar_pvp_forecasting'
code .
```

Alternatively, on GitHub select **Code > Download ZIP**, extract it, and use **File > Open Folder** in VS Code. Open the folder containing `pyproject.toml`. If you already have the project folder, open it directly and skip cloning.

### 1.6 Obtain the datasets separately

**Cloning the repository does not download the data.** `.gitignore` excludes `Data/`, `Resources/`, `Paper/`, run outputs, and reference PDFs. Obtain the processed `Data/` folder from the project owner or lab's shared archive and extract it into the project root.

There is no dataset downloader or raw-to-feature conversion script in this checkout. Raw spreadsheets or the NIST ZIP alone are insufficient for this pipeline. Obtain the processed Parquet files listed below; do not rename unrelated raw files to these names.

The minimum data layout for both sites is:

```text
Solar_pvp_forecasting/
  Data/
    features_ssp_2025/
      dataset.parquet
    features_nist_ground_2017/
      ac_power.parquet
      ground_radiation_wm2.parquet
      pv_module_temperature.parquet
```

| Site | File under `Data/` | Required columns |
|---|---|---|
| SSP | `features_ssp_2025/dataset.parquet` | `datetime`, `ac_power`, `roof_radiation_wm2`, `pv_module_temperature` |
| NIST | `features_nist_ground_2017/ac_power.parquet` | `datetime`, `ac_power` |
| NIST | `features_nist_ground_2017/ground_radiation_wm2.parquet` | `datetime`, `ground_radiation_wm2` |
| NIST | `features_nist_ground_2017/pv_module_temperature.parquet` | `datetime`, `pv_module_temperature` |

The loader expects naive local timestamps: SSP uses Bangladesh time; NIST uses `America/New_York`, including daylight saving. SSP power is read in kW; NIST power is read in W and converted to kW. These mappings and site ratings are defined in `pvf/data/sites.py`.

Only the selected site's files are needed to build its data bundle. However, `pvf.check_environment()` checks **both sites**, so missing files for the other site still produce FAIL lines.

Optionally obtain `Resources/`, `ref/` PDFs, and `Paper/` from the lab for reports and reference material. They are not required to train a model. `Data/cache/` is generated automatically.

### 1.7 Create the environment and download packages

From the project root, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1
```

This creates `.venv`, upgrades pip, downloads the selected PyTorch build, installs packages pinned in `requirements/lock-windows.txt`, installs `pvf` in editable mode, and prints the environment check. Editable installation means changes in `pvf/` are used without reinstalling the project.

The Windows script currently chooses:

| Detected driver major version | Selected requirements file |
|---|---|
| 580 or newer | `requirements/torch-cu130.txt` |
| 528 through 579 | `requirements/torch-cu126.txt` |
| Older, or no detected NVIDIA driver | `requirements/torch-cpu.txt` |

These are the script's selection rules, not a guarantee that every GPU supports every build. Confirm the result with the check below. To force a build, run **one** of:

```powershell
# NVIDIA CUDA 13.0 build
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1 -Cuda cu130

# NVIDIA CUDA 12.6 build
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1 -Cuda cu126

# CPU build
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1 -Cuda cpu
```

The requirements include PyTorch, NumPy, pandas, PyArrow, pvlib, mambapy, plotting tools, JupyterLab, ipykernel, and the foundation-model libraries. You do not need to install these individually, install Anaconda, or download Jupyter separately.

Let package installation finish. The setup script stops on installation errors or a failed readiness check. Verify explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import pvf; pvf.check_environment()"
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('Runtime:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

Expect `READY` when the required packages and data files are present, on either CPU or GPU. For GPU use, also expect `GPU available: True`. The trainer automatically uses CUDA when available and CPU otherwise; CPU runs disable mixed precision. Resolve package and data failures before experimenting.

On Windows, LSTM/GRU layers use PyTorch's native implementation on the selected device to avoid a shutdown crash observed with the pinned cuDNN runtime. They still train on GPU when available, with a possible performance cost. Other layers retain cuDNN acceleration.

You do not need to activate `.venv` when using its full executable path, as these commands do.

### 1.8 Select the notebook kernel

1. Open `notebooks/run_experiment.ipynb` in VS Code.
2. Click **Select Kernel** at the top right.
3. Choose **Python Environments**, then the project's `.venv` interpreter: `.venv\Scripts\python.exe`.
4. Check it in a notebook cell:

```python
import sys
import pvf

print(sys.executable)
print(pvf.__file__)
```

Both paths should point into this project. Selecting a terminal interpreter and selecting a notebook kernel are separate actions. If `.venv` is missing from the picker, reload the VS Code window and try again. If necessary, register it:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name pvf --display-name "Python (pvf)"
```

Then select **Python (pvf)**. As an alternative to VS Code notebooks, start JupyterLab:

```powershell
.\.venv\Scripts\jupyter.exe lab
```

Use the local URL printed in that terminal.

### 1.9 Download pretrained weights only if needed

LSTM, GRU, Transformer, Mamba, and other models trained by this project do not need external pretrained weights. The three foundation-model wrappers download weights on first construction if the local copy is missing:

| Model name | Weight source | Local folder under `Data/pretrained/` |
|---|---|---|
| `chronos2` | [Amazon Chronos-2](https://huggingface.co/amazon/chronos-2) | `amazon--chronos-2/` |
| `timesfm` | [Google TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) | `google--timesfm-2.5-200m-pytorch/` |
| `tirex` | [NX-AI TiRex](https://huggingface.co/NX-AI/TiRex) | `NX-AI--TiRex/` |

To download ahead of time, run this in a notebook cell with internet access:

```python
from pvf.models.architectures.foundation_models import (
    WEIGHTS_REPOSITORIES,
    local_weights,
)

for repository in WEIGHTS_REPOSITORIES.values():
    print(local_weights(repository))
```

The helper creates a `.complete` marker after each successful download. For an offline machine, copy the full pretrained folders, including hidden `.complete` files, from a working installation. Do not create the markers yourself to bypass incomplete downloads. These wrappers perform zero-shot evaluation; they do not fine-tune the downloaded models.

## 2. What the files contain

### 2.1 Top-level files and folders

| Path | Contents and purpose |
|---|---|
| `README.md` | Project overview, conventions, model contract, and extension examples |
| `notebooks/run_experiment.ipynb` | Main interactive workflow: choose a model, prepare data, inspect inputs, train, plot, compare, and tune |
| `setup_venv.ps1`, `setup_venv.sh` | Windows and Linux environment setup |
| `pyproject.toml` | Package metadata and editable-install configuration; dependency pins live in `requirements/` |
| `requirements/lock-windows.txt`, `lock-linux.txt` | Pinned Python dependencies for each operating system |
| `requirements/torch-*.txt` | Separate pinned CPU / CUDA PyTorch builds |
| `Data/` | Processed datasets, optional raw archives, generated cache, and optional pretrained weights |
| `pvf/` | Data preparation, model definitions, training, evaluation, and experiment orchestration |
| `docs/training_recipes.md` | Reference-paper settings and documented implementation differences |
| `tools/run_all_models.py` | Batch experiments and Markdown/PDF report generation |
| `tools/markdown_to_pdf.py` | Converts Markdown reports to PDF |
| `runs/` | Individual experiment configurations, checkpoints, predictions, and metrics |
| `results/` | Experiment CSV log and batch manifests |
| `Resources/` | Lab reference material and generated batch reports/figures |
| `ref/README.md`, `ref/` | Reference-paper links and separately shared PDFs |
| `Paper/` | Separately shared manuscript material |
| `.gitignore` | Defines local data, outputs, environments, and documents excluded from Git |

### 2.2 Python source map

| File or folder | What it controls |
|---|---|
| `pvf/__init__.py` | Public helpers available through `import pvf` |
| `pvf/data/sites.py` | SSP/NIST file paths, source columns, units, coordinates, timezones, and inverter ratings |
| `pvf/data/site_record.py` | Reads Parquet, handles timestamps, resamples, computes clear sky, caches site records, and computes smart persistence |
| `pvf/data/features.py` | Feature names, physical scaling, calendar features, and which features are allowed in the future inputs |
| `pvf/data/splits.py` | Weekly blocked splits, explicit month splits, and holdout months |
| `pvf/data/windows.py` | History/target windows, missing-data masks, and split-boundary rules |
| `pvf/data/pipeline.py` | `DataConfig`, `DataSpec`, dataset bundles, data loaders, and sample batches |
| `pvf/model_inputs/input_adapter.py` | Runs named input steps and provides `trace()` / `plot()` inspection |
| `pvf/model_inputs/adapter_steps.py` | Input transformations used by different architectures |
| `pvf/models/base.py` | Model registry, `build_model()`, shared model interface, and optional persistence residual wrapper |
| `pvf/models/architectures/baselines.py` | `smart_persistence`, `lstm`, and `gru` |
| `pvf/models/architectures/transformers.py` | `patch_tst`, `i_transformer`, and `time_xer` |
| `pvf/models/architectures/state_space.py` | `mamba_temporal`, `s_mamba`, and `power_mamba` |
| `pvf/models/architectures/hdcf_mamba.py` | HDCF-Mamba architecture |
| `pvf/models/architectures/graph_mamba.py` | Graph-Mamba; current graph nodes represent variables |
| `pvf/models/architectures/rfkan.py` | Recurrent Fourier KAN |
| `pvf/models/architectures/probabilistic.py` | `mtsa_qrn` and `dt_dann` |
| `pvf/models/architectures/foundation_models.py` | Chronos-2, TimesFM, TiRex, and local weight downloads |
| `pvf/models/architectures/fusion_example.py` | Worked iTransformer + HDCF-Mamba fusion |
| `pvf/models/architectures/__init__.py` | Imports architectures so their registered names become available |
| `pvf/models/blocks/` | Shared heads, normalization, Mamba, graph, and other layers |
| `pvf/training/paper_recipes.py` | Per-model paper presets, recommended lookback durations, and notes |
| `pvf/training/trainer.py` | `TrainingRecipe`, device selection, training, early stopping, checkpointing, and prediction |
| `pvf/training/losses.py` | Masked point and quantile losses |
| `pvf/training/optimizers.py` | Optimizer and scheduler construction |
| `pvf/experiment.py` | `ExperimentConfig`, `run_experiment()`, saving results, and reloading models |
| `pvf/search.py` | Saved hyperparameter trials ranked by validation performance |
| `pvf/evaluation/metrics.py` | Overall/per-horizon metric calculations and metric descriptions |
| `pvf/evaluation/comparison.py` | Run comparison tables and diagnostic plots |
| `pvf/evaluation/conformal.py` | Conformal interval calibration helpers |
| `pvf/workspace.py` | Environment readiness check |

For normal experiments, edit the notebook's **CHOOSE** cells. Change source files only when adding or modifying a feature, architecture, input transformation, loss, or other implementation.

The flow is:

```text
DataConfig -> data bundle -> input adapter -> model
                                    + TrainingRecipe
                                    -> run_experiment -> saved results
```

## 3. Run experiments

### 3.1 Run a small first experiment

Use this self-contained example in a new notebook with the `.venv` kernel. It uses SSP data, a small LSTM, one epoch, and 5% of training windows to check the complete workflow. It still evaluates the full validation/test splits, so it is a setup check rather than a competitive result.

```python
import pvf
from pvf.training.trainer import seed_everything

data_config = pvf.DataConfig(
    site="ssp",
    resolution="30min",
    lookback_steps=48,       # 24 hours of history
    horizon_steps=48,        # jointly forecast 0.5 to 24 hours ahead
)
recipe = pvf.TrainingRecipe(
    n_epochs=1,
    batch_size=16,
    train_fraction=0.05,
    mixed_precision="none",
    num_workers=0,
    seed=0,
)
config = pvf.ExperimentConfig(
    model_name="lstm",
    model_hyperparameters={"hidden_size": 32, "n_layers": 2},
    data=data_config,
    recipe=recipe,
    run_name="lstm_ssp_setup_check",
)

seed_everything(recipe.seed)  # seed before run_experiment constructs the model
result = pvf.run_experiment(config)
print(result.run_dir)
display(result.summary())
```

Use `site="nist"` for the other site. A successful run prints epoch progress, validation/test metrics, and the saved run folder. `holdout` appears only when configured and nonempty.

### 3.2 Use the main experiment notebook

Open `notebooks/run_experiment.ipynb` and work through its numbered sections. `Shift+Enter` executes a cell. For your first session, run cells individually so you can inspect choices before training. **Run All** also executes training and scoring, and executes search if `run_search=True`.

| Notebook section | Action |
|---|---|
| Imports and 0 Machine check | Import `pvf` and inspect readiness |
| 1 Model | Inspect `describe_models()` / `describe_paper_recipes()`; choose `model_name`, preset use, and constructor arguments |
| 2 Data | Choose site, features, lookback, horizon, and split; build the bundle and a fresh model |
| 3 Input adapter | Inspect how the standard batch becomes the selected architecture's input |
| 4 Training recipe | Choose training settings and a descriptive `run_name` |
| 5 Train and score | Run the experiment and view the metric table |
| 6 Diagnose | View losses, forecast examples, error versus lead time, and error by hour |
| 7 Compare runs | Compare saved runs for this site |
| 8 Hyperparameter search | Optional search; disabled by default |
| 9 Fusing models | Read the fusion example instructions |
| 10 Reload a saved model | Load the completed run's model |

Changing model names or hyperparameters requires rerunning the model-construction cell. Changing data settings requires rebuilding both the data bundle and the model, then rerunning the recipe/configuration and training cells. Restart the kernel and run from the top when uncertain about stale state. After source edits, restart the kernel so imported classes refresh.

**Create a fresh model for every independent run.** Rerunning only section 5 passes the existing model object back to the trainer, starting from its current weights.

### 3.3 Choose the data and evaluation protocol

The defaults are SSP, 30-minute resolution, 128 history steps (64 hours), and 48 forecast steps (24 hours). Model paper presets can replace the lookback. Duration equals the number of steps multiplied by the resolution.

List the available features:

```python
display(pvf.describe_features())
```

`history_feature_names` selects observations available up to the forecast origin. `future_feature_names` selects values known for target times: clear sky, calendar features, and yesterday's power. Future measured irradiance or temperature is unavailable in a real forecast. The explicit `*_oracle` features use those measurements as perfect forecasts and must be labelled as upper-bound experiments.

With the default `ac_power_yesterday` future feature, the horizon must be at most one day. For longer horizons, remove it and choose other valid future features.

The default split repeats four training weeks, one validation week, and one test week. It covers seasons in all splits and purges training histories containing evaluation timesteps, but nearby blocks are correlated. For a chronological experiment with December reserved:

```python
split = pvf.SplitConfig(
    mode="months",
    train_months=[1, 2, 3, 4, 5, 6, 7, 8],
    validation_months=[9],
    test_months=[10, 11],
    holdout_months=[12],
)
data_config = pvf.DataConfig(site="ssp", split=split)
data_bundle = pvf.build_data_bundle(data_config)
display(pvf.describe_data_bundle(data_bundle))
```

Use this `data_config` in the experiment you run; defining it alone does not change an existing configuration. The month lists apply to the available site year, rather than specifying year ranges.

### 3.4 Start from a reference-paper recipe

```python
from pvf.training.trainer import seed_everything

model_name = "hdcf_mamba"
preset = pvf.paper_recipe(model_name)
print(preset.source)
print(*preset.notes, sep="\n")

data_config = pvf.DataConfig(
    site="ssp",
    resolution="30min",
    lookback_steps=preset.lookback_steps("30min"),
)
recipe = preset.recipe
recipe.seed = 0
recipe.time_limit_minutes = 15

config = pvf.ExperimentConfig(
    model_name=model_name,
    model_hyperparameters=dict(preset.model_hyperparameters),
    data=data_config,
    recipe=recipe,
    run_name="hdcf_mamba_ssp_paper_seed0",
)
seed_everything(recipe.seed)
result = pvf.run_experiment(config)
display(result.summary())
```

Change `model_name` to another registered name to use its preset. Read `preset.notes` and `docs/training_recipes.md`: presets contain documented substitutions, and the local data/horizon may differ from the original paper.

The main notebook currently sets `recipe.n_epochs = 100` after loading the preset. Remove or edit that override when you want to preserve the preset's epoch count. Any time cap or other change also becomes part of your experimental protocol.

Useful recipe controls:

| Field | Meaning |
|---|---|
| `n_epochs` | Maximum training epochs |
| `early_stopping_metric` | `validation_loss`, `nmae`, or `nrmse`; lower is better |
| `early_stopping_patience` | Epochs without sufficient improvement; `None` disables early stopping |
| `time_limit_minutes` | Training cap checked after each epoch; downloads and final scoring are outside it |
| `batch_size` | Windows per batch; reduce if GPU memory is exhausted |
| `gradient_accumulation_steps` | Accumulate several batches before an optimizer update |
| `loss`, `loss_options` | Point-loss selection/options; quantile models use pinball automatically |
| `optimizer`, `learning_rate`, `weight_decay` | Optimization settings |
| `scheduler`, `scheduler_options`, `warmup_epochs` | Learning-rate schedule |
| `mixed_precision` | `bf16`, `fp16`, or `none`; bf16 requests automatically fall back to fp16 on unsupported GPUs, and CPU always uses float32 |
| `train_fraction` | Share of training windows; keep `1.0` for a full-data run |
| `seed` | Training randomness and subsampling seed |
| `num_workers` | Keep `0` for the Windows notebook workflow |
| `initialize_from_run`, `freeze_parameter_prefixes` | Start from saved weights and optionally freeze matching parameter names |

The trainer restores the best checkpoint according to the selected validation metric. A fixed epoch count still retains the best epoch's weights.

For repeatable initialization, call `seed_everything(seed)` **before constructing the model**. The trainer also seeds training, but model construction happens earlier. In the main notebook, place this call immediately before `pvf.build_model(...)`, using the same seed later assigned to the recipe. Exact results can still vary between hardware/software environments.

### 3.5 Inspect results and saved files

```python
display(result.summary())
pvf.plot_training_history(result.run_dir);
pvf.plot_forecast_examples(result.run_dir, split="test", n_examples=3);
pvf.plot_per_horizon([result.run_dir], metric="nrmse_pct", split="test");
pvf.plot_error_by_hour([result.run_dir], split="test");
```

Scores use measured daytime targets and normalize power errors by the site's inverter AC rating. For example, nMAE of 5% means mean absolute error is 5% of rated power, not 5% of the measured power at each timestamp.

| Score | How to read it |
|---|---|
| `nmae_pct`, `nrmse_pct` | Lower is better; nRMSE penalizes large errors more |
| `nbias_pct` | Near zero is better; positive means overprediction |
| `rmse_skill_vs_persistence` | Positive means better than smart persistence |
| `r2` | Higher is better; 1 is perfect |
| `ramp_mae_pct` | Lower error on rapid changes is better |
| `crps`, pinball scores | Distributional accuracy for quantile forecasts; lower is better |
| Interval coverage/width | Interpret together: wider intervals can increase coverage |

Read skill together with nMAE/nRMSE: smart persistence is weak for long horizons starting at night. Full definitions are in `pvf/evaluation/metrics.py`.

Each completed run writes `runs/<site>/<run_name>/<timestamp>/`:

| File | Contents |
|---|---|
| `config.yaml` | Data choices, architecture arguments, training recipe, and evaluation splits |
| `best_model.pt` | Best weights for trained models |
| `trained_model.pt` | Whole trained model when serialization succeeds; reload falls back to config + weights if absent |
| `metrics.json` | Training history, overall metrics, and per-step metrics |
| `predictions_validation.npz`, `predictions_test.npz`, optional `predictions_holdout.npz` | Forecasts, targets, masks, persistence reference, and forecast-origin indices/timestamps |

Zero-shot models and smart persistence have no training checkpoint. Their config and metrics are still saved. The experiment also appends a row to `results/experiment_log.csv`.

Reload a run with:

```python
trained_model = pvf.load_trained_model(result.run_dir)
# In a later session, pass the actual saved folder path instead of result.run_dir.
```

Keep the run folder, matching source code, and dataset version together. Foundation-model reloads also need the external weights. Run outputs are excluded from Git, so share them separately.

### 3.6 Compare runs and tune hyperparameters

```python
comparison = pvf.compare_runs(site="ssp", split="validation", latest_only=False)
display(comparison)
```

By default, `compare_runs()` shows the latest run per `run_name` and sorts by nRMSE. Use distinct names for changed settings and seeds, or `latest_only=False` to see all runs.

Tune on validation, then inspect the test set after selecting the settings. For an optional search after constructing `config`:

```python
from copy import deepcopy

search_config = deepcopy(config)
search_config.run_name += "_tuning"
search_config.evaluation_splits = ["validation"]
search_results = pvf.search_hyperparameters(
    search_config,
    {"recipe.learning_rate": [1e-4, 3e-4, 1e-3]},
    n_trials=3,
)
display(search_results)
```

Each trial is saved and ranked by validation nRMSE. Model arguments use keys such as `model.model_size` for architectures that accept that argument; LSTM instead accepts `model.hidden_size`. Inspect `pvf.describe_models()` before changing constructor arguments. Transfer the chosen settings to a fresh final configuration and run it with the intended test/holdout evaluation splits. The current search helper seeds trial ordering and training; it does not explicitly reseed each model before construction.

### 3.7 Run a batch from the terminal

The batch script loads paper presets, trains/evaluates the selected models, and generates a Markdown/PDF report. Start with a small subset:

```powershell
.\.venv\Scripts\python.exe tools/run_all_models.py --site ssp --models lstm,patch_tst,hdcf_mamba --time-limit 15
```

Run every model with a preset:

```powershell
.\.venv\Scripts\python.exe tools/run_all_models.py --site ssp --time-limit 15
```

Use `--site nist` for NIST. All-model runs include the three foundation models, so allow for weight downloads, long input histories, and inference time. The time cap applies to each model's training and is checked at epoch boundaries; it is not a total wall-clock deadline.

The script intersects test forecast origins across the selected lookbacks so every model in that batch is scored on the same test windows. Separate batches with different model subsets may have different intersections. Ordinary notebook runs do not automatically align test windows across different lookbacks.

Outputs include:

```text
results/batch_ssp.json
Resources/Experiment_Report_SSP_<date>.md
Resources/Experiment_Report_SSP_<date>.pdf
Resources/Experiment_Report_SSP_<date>_figures/
```

The report includes test rankings, per-horizon scores, recipes, training histories, and failures. Rebuild it without retraining:

```powershell
.\.venv\Scripts\python.exe tools/run_all_models.py --site ssp --report-only
```

The manifest is saved as models complete, but a new batch invocation starts a new manifest; it does not automatically resume unfinished models. Preserve `results/batch_<site>.json` before starting another batch for that site. Reports with the same site and date reuse the same filenames. Individual timestamped run folders remain available.

### 3.8 Use a consistent research protocol

1. Record the site, data version, resolution, history/future features, horizon, split, and any oracle inputs.
2. Select hyperparameters using validation data. Keep test/holdout data out of that selection process.
3. Keep test origins aligned across compared models, particularly when lookbacks differ. Use a single batch for the alignment provided by the batch script.
4. Distinguish a comparison using each model's paper recipe from an ablation using a shared recipe and data configuration.
5. Use several seeds for final trainable-model comparisons and report variation. Seed initialization before model construction for each run.
6. For a reserved holdout, set `evaluation_splits=["validation", "test"]` during development and include `"holdout"` only for the final evaluation. The default evaluation list includes holdout whenever it exists.
7. Save the run folders and identify the source revision with `git rev-parse HEAD`. `config.yaml` does not automatically record the Git revision, complete package environment, or dataset checksum.

## 4. Troubleshooting

| Problem | What to check |
|---|---|
| `git`, `py`, or `code` is not recognized | Reopen the terminal after installation; verify PATH. Open the folder through VS Code's menu if only `code` is missing |
| `ModuleNotFoundError: pvf` | Select the project `.venv`; rerun `.\.venv\Scripts\python.exe -m pip install --no-deps -e .` from the root |
| A dependency fails to install | Read the first pip error, verify Python 3.11 and network access, and rerun the failed requirements step. Avoid silently substituting versions if reproducibility matters |
| `.venv` came from another machine | Recreate the environment locally using the setup script; copying a virtual environment is unreliable |
| `nvidia-smi` fails | Verify the NVIDIA driver installation, restart if required, or explicitly use CPU |
| `nvidia-smi` works but PyTorch reports no GPU | Check the notebook interpreter and PyTorch build; a CPU wheel does not enable CUDA. After correcting the build, restart the kernel |
| GPU out of memory | Reduce batch size; then consider smaller model settings or lookback. Record these changes and keep comparison conditions consistent |
| GPU lacks bf16 support | bf16 requests automatically use fp16; select `"none"` if full precision is needed |
| Windows process exits with `-1073740791` after printing results | Previously observed in GPU training with the pinned runtime. The current recurrent layers bypass cuDNN on Windows, and the CPU/GPU suites exit cleanly with this workaround. If it recurs, check that the kernel loaded the updated code and investigate the runtime/driver environment |
| Missing Parquet file | Check the exact paths in section 1.6; cloning Git does not supply `Data/` |
| Missing column or timestamp error | Confirm that you have the expected processed dataset, with the columns and timestamp convention in section 1.6 |
| `no training samples` | Increase `split.block_days`, shorten the lookback, or choose a suitable month split. Long histories must fit in usable training regions |
| Future feature rejected | Use only known-in-advance inputs; use an explicitly labelled oracle experiment if appropriate |
| Results do not reflect changed settings | Rebuild the bundle/model and experiment config, or restart the kernel and rerun in order |
| No holdout metrics | Set nonempty holdout months, include `"holdout"` in evaluation splits, and confirm that the dataset has valid windows there |
| Foundation-model download fails | Check access to its Hugging Face repository or copy a complete local weight folder from a working machine |
| Report-only command fails | Check that `results/batch_<site>.json` exists and its referenced run folders are still accessible |

## 5. Linux setup equivalent

With Python 3.11 and VS Code already installed, install Git using the distribution's package manager. For Ubuntu/Debian:

```bash
sudo apt update
sudo apt install git python3.11-venv
```

Install an NVIDIA driver suitable for the distribution and GPU if using CUDA, and check `nvidia-smi`. Obtain the repository and processed `Data/` folder as above, then run from the project root:

```bash
bash setup_venv.sh
# Or explicitly select one build:
# bash setup_venv.sh cu126
# bash setup_venv.sh cpu

.venv/bin/python -m pip check
.venv/bin/python -c 'import pvf; pvf.check_environment()'
```

The Linux script uses `requirements/lock-linux.txt`; its automatic cu126 selection starts at driver major 525. Select `.venv/bin/python` as the VS Code notebook kernel. The Python notebook examples are the same. Terminal commands use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

For Linux JupyterLab or a batch:

```bash
.venv/bin/jupyter lab
.venv/bin/python tools/run_all_models.py --site ssp --models lstm,patch_tst --time-limit 15
```
