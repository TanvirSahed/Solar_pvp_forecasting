# pvf — PV AC-power forecasting research framework (SET Lab, MIST)

Re-implement, compare, rewire and fuse the forecasting models from
`Resources/SET_Lab_Dl_Model_Selection.pdf` on the NWPGCL Sirajganj (SSP, 2025) and NIST ground-array (2017) data.
PyTorch 2.14 (CUDA 13.0), Python 3.11.

## 1. Setup on a new machine (e.g. a colleague's PC)

Needs: an NVIDIA GPU + driver, git, and either **Python 3.11** or **Docker**.

```powershell
git clone <repo-url> powerForecasting_model
cd powerForecasting_model
# copy the Data/ folder (and Resources/, ref/ if wanted) into this directory
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1     # Linux: bash setup_venv.sh
```
The setup script installs the exact package versions from `requirements/lock-*.txt` and picks the PyTorch CUDA
build from the installed driver: cu130 for driver 580 or newer, cu126 for 528 or newer, otherwise CPU.
It finishes with `pvf doctor`, which must print **READY**. It checks Python, GPU, packages, and that every data file
is byte-identical to the checksums in `configs/data_manifest.json`.

**With Docker instead** (Docker Desktop with WSL2 on Windows, or Docker + NVIDIA Container Toolkit on Linux):
```bash
docker compose build                    # older driver (< 580): TORCH_CUDA=cu126 docker compose build
docker compose run --rm pvf python -m pvf doctor
docker compose run --rm pvf python -m pvf bench -x configs/experiments/round1.yaml site=ssp
docker compose up lab                   # JupyterLab -> http://localhost:8888  (token: setlab)
```

### Running the same experiments on both machines

| Who | Step |
|---|---|
| you | put the experiment in `configs/experiments/<name>.yaml` (or a model YAML), then `git commit` and `git push` |
| colleague | `git pull`, then `python -m pvf bench -x configs/experiments/<name>.yaml site=ssp` |
| either | `python -m pvf rerun runs/<site>/<model>/<timestamp>` repeats any run from its folder (send the folder; it is small) |

Every run folder contains `config.yaml` (the full resolved config) and `provenance.json` (git commit, data
checksums, GPU, versions). It also contains `git_diff.patch` if the code had uncommitted changes, so `rerun` tells
you which commit to check out. With `deterministic=true`, a rerun on the same GPU type is bit-identical (verified).
Different GPUs agree only to floating-point noise. If the data ever changes, run `python -m pvf data-manifest`
and commit the result; `doctor` on the other machine will then flag stale copies.
Foundation-model weights (Chronos-2, TimesFM, TiRex) download from Hugging Face on first use. To avoid that,
copy `%USERPROFILE%\.cache\huggingface` over as well.

## 2. Data

`pvf.data.loader.load_site(site)` builds the uniform record from `Data/features_*`, cached in `Data/cache/`:

| | NIST (Gaithersburg, 260 kWac) | SSP (Sirajganj, 2.33 MW inverter block) |
|---|---|---|
| native | 1-min, P in W | P/T 5-min (kW), G 30-min |
| used | P [kW], G [W/m²], T [°C] → 30-min interval mean | same |
| timezone | **America/New_York (DST)** | GMT+6 |

Added features: Ineichen clear-sky GHI `Gcs` (pvlib), calendar sin/cos, `P(t+h−24h)`, and smart persistence
`s(t,h) = P(t)·Gcs(t+h)/Gcs(t)`. Inputs are normalised physically (P/P_rated, G/1000, (T−25)/25), so there is
no fitted scaler to leak. Metrics on normalised P are therefore nMAE/nRMSE as a fraction of capacity.

> NIST timestamps follow local clock time **with** daylight saving: the mean irradiance peak sits at ~12:05 in
> January and ~12:54 in June. A fixed GMT−5 would shift clear-sky and smart persistence by 1 h from March to November.

**Splits** (`configs/base.yaml → data.split`). The default is `blocked`: weekly blocks cycle
train×4 / val / test, so every season is in every split. A sample's 48 targets lie inside one block, and training
input windows never contain val/test timesteps (purge). Use `holdout_months: [12]` for a final untouched test,
or `mode: months` for a pure chronological split.

## 3. Models (`python -m pvf list`)

| Report row | Registry name | Source |
|---|---|---|
| ref | `smart_persistence` | clear-sky persistence |
| 1 | `lstm`, `gru`, `tcn`, `mlp`, `dlinear` | controls |
| 2 | `patchtst` (+ ACI via `pvf.eval.conformal.aci`) | F01, R07 |
| 3 | `itransformer` | F02 |
| 4 | `timexer` | F03 |
| 5 | `hdcf_mamba` | R06 §3 (MSFEM + single Mamba + temporal concat + 2-step linear decoder) |
| 6 | `powermamba` | R19 §IV (RevIN, decomposition, Mamba ∥ iMamba, external-forecast appending) |
| — | `mamba` (Concat-Mamba), `smamba` | F04, F06 |
| 7–9 | `timesfm`, `chronos2`, `tirex` (zero-shot) | R14, R12, R13/F07 |
| 10 | `rfkan` | R16 Eq. 3–8 |
| 11–12 | `graph_mamba` (GCN/GAT, adjacency none/physics/learned/mixed α_t) | R21–R23, report §3 |
| 13 | `dt_dann` (causal-CNN → Transformer, GRL domain classifier) | R18 |
| 14 | `mtsa_qrn` (+ CQR via `pvf.eval.conformal.cqr`) | R08 §3 |
| 15 | `pvf.ensemble.selector` (static-best, inverse-error, LinUCB; the RL agents plug in here) | R24–R26, R35 |
| new | `fusion` (mean / gate / stack over any models, deep supervision) | — |

The graph model's nodes are currently the physical variables {P, G, T, Gcs} with a physics prior, because each
dataset holds a single inverter. Override `GraphMamba.build_nodes` when per-inverter data (Inv1…Inv4) arrive.

## 4. Running experiments

```bash
python -m pvf data site=nist                                   # split summary
python -m pvf train -m patchtst site=nist                      # one run
python -m pvf train -m graph_mamba model.conv=gcn model.adj=learned tag=gcn   # any key overridable
python -m pvf train -m hdcf_mamba model.anchor=sp              # learn the residual over smart persistence
python -m pvf bench -x configs/experiments/round1.yaml site=ssp
python -m pvf bench -x configs/experiments/graph_ablation.yaml site=ssp
tensorboard --logdir runs
python -m pytest -q                                            # leakage / split / conformal tests
```
Each run writes `runs/<site>/<model>[_tag]/<timestamp>/` containing `config.yaml`, `best.pt`, `metrics.json`
(overall + per-horizon, skill vs smart persistence, ramp error, CRPS/PICP for quantile models) and
`pred_<split>.npz`. Bench runs also append to `results/<site>_<experiment>.csv`.

## 5. Changing architectures

The model contract lives in `pvf/models/base.py`: `forward(batch) -> {"pred": [B,H] | [B,H,Q], "aux_loss"?: scalar, ...}`.
`batch` = `x_hist [B,L,C_h]`, `x_fut [B,H,C_f]`, `y`, `mask`, `sp`, `t0`.

* **New model:** a new file in `pvf/models/zoo/`, decorate it with `@register("name")`, add it to `zoo/__init__.py` and `configs/models/name.yaml`.
* **Building blocks** (`pvf/models/layers/`): `MambaMixer`/`MambaLayer` (pure-PyTorch selective scan; `mamba_ssm` CUDA kernel used automatically on Linux if installed), `DenseGCN`, `DenseGAT`, `AdjacencyLearner`, `FourierKAN`, `RevIN`, `SeriesDecomp`, `PatchEmbed`, `HorizonHead`, `grad_reverse`.
* **New connections:** any model can reuse another's encoder (for example `MambaTemporal.encode`); `HorizonHead(use_fut=True)` gives every backbone the same access to future clear-sky/NWP.
* **Fusing models** from config only: see `configs/models/fusion_gate.yaml`.
* **Physics anchoring:** `model.anchor=sp` wraps any model as `pred = smart_persistence + f(x)`.
* **Extra losses / domain adaptation:** return `aux_loss`, or `deep_preds`, and set `data.adapt_months` to get `batch["tgt"]`.

## 6. Papers

`ref/` (transferred with the data, not in git) holds the selection-report papers R01–R35 and the foundational
architecture papers. `ref/README.md` is the link list. `tools/fetch_refs.py` can rebuild the folder if needed.
