# How the reference papers trained each model

As of 2026-10-18. This covers the 15 models in `Resources/SET_Lab_Dl_Model_Selection.pdf`, read from the PDFs in `ref/`.

**Bottom line.**
- **Training recipes are thin.** Only 4 papers give a complete one: R20, R6, R7 and R3 (the last for fine‑tuning only). Most say "Adam + MSE + early stopping" and tuned everything else (Optuna, grid search or Nixtla) without publishing the values.
- **The PV papers converge on one recipe:** Adam(W), lr 1e‑4 to 1e‑3, MSE loss, early stopping on validation, a plateau or one‑cycle schedule, d_model 128–256, dropout 0.1–0.2, and a lookback of 1–3 days or more.
- **Foundation models are best used zero‑shot with covariates and a long context.** Fine‑tuning them is not clearly better (R3).
- **Watch what is being compared.** Several PV results rest on small or unusual test sets (RFKAN: one test day per season). Evaluation horizons also range from 1 h to 24 h, so headline numbers aren't comparable across papers.

"Not reported" means the paper does not say. Lookback and horizon are in steps at the paper's resolution.

## Coverage

| # | Model (selection report) | Papers read | Not available (paywalled) |
|---|---|---|---|
| 1 | LSTM / GRU | R2, R20, (R16, R32 baselines) | – |
| 2 | PatchTST + ACI | R7, R2, R20, F01 (original) | – |
| 3 | iTransformer | R20, R6, F02 (original) | – |
| 4 | TimeXer | R2, F03 (original) | – |
| 5 | HDCF‑Mamba | R6 | – |
| 6 | PowerMamba | R19, R20 | – |
| 7 | TimesFM | R14, R2, R4 | – |
| 8 | Chronos‑2 | R12, R3, R4 | – |
| 9 | TiRex‑2 | R13, R3 | – |
| 10 | RFKAN | R16 | – |
| 11 | DEST‑GNN | – | R21, R22, R23 |
| 12 | DT‑informed GAT/GCN + Mamba | inherits R6, R18 | R21–R23 |
| 13 | DT Transformer + CNN + DANN | R18 | – |
| 14 | MTSA‑QRN + CQR | R8 | – |
| 15 | RL selector / fusion | R25, R32, R33 | R24, R26, R35 |

## Summary

| Model | Data (paper) | Lookback → horizon | Optimizer, lr, schedule | Batch / epochs / early stop | Loss | Key hyperparameters |
|---|---|---|---|---|---|---|
| LSTM (R20) | EIA‑930 hourly load, 6 US grids | 240 h → 24–168 h | AdamW 1e‑3, wd 1e‑4, OneCycleLR | ≤200 ep, patience 30 on val MAPE | MSE | 2‑layer **bidirectional**, hidden 256, attention pooling |
| LSTM (R2) | ERCOT 5‑min solar/wind/load | 512 → 288 (24 h) | Adam, tuned (Nixtla) | early stop, values not reported | quantile | not reported |
| PatchTST (R7) | 5 kWp rooftop PV, hourly, 5 yr | 48 → 1 | not reported | batch 48, **10 ep** | MSE | d_model 128, 4 heads, patch 16, stride 8, dropout 0.1 |
| PatchTST (R20) | as R20 above | 240 h → 24–168 h | AdamW 1e‑3, OneCycle | ≤200 ep, patience 30 | MSE | d_model 256 |
| PatchTST (F01) | long‑term benchmarks | 336 → 96–720 | not stated in the paper | not stated in the paper | MSE | 3 layers, 16 heads, D 128, FF 256, dropout 0.2, patch 16/8 |
| iTransformer (R20) | as R20 above | 240 h → 24–168 h | AdamW 1e‑3, OneCycle | ≤200 ep, patience 30 | MSE | d_model 512 (256 keeps the weather gain) |
| iTransformer (F02) | benchmarks | 96 → 96–720 | Adam {1e‑3, 5e‑4, 1e‑4} | batch 32, **10 ep** | MSE | 2–4 blocks, D {256, 512} |
| TimeXer (F03) | benchmarks + EPF | 96 → 96–720 | Adam 1e‑4 | 10 ep + early stop | MSE | 1–3 blocks, d_model {128, 256, 512}, patch 16 (long) / 24 (short) |
| TimeXer (R2) | ERCOT 5‑min | 512 → 288 | Adam, tuned | early stop | quantile/dropout | ~19M params |
| HDCF‑Mamba (R6) | 4 PV plants, 15‑min | up to 720 → 24–512 | **Adam 1e‑4** (β 0.9/0.999), **ReduceLROnPlateau ×0.5** | Optuna batch, **patience 4** | MSE | d_model 128, d_state **32**, expand 2, d_conv 4, dropout 0.1, 1 Mamba block |
| PowerMamba (R19) | ERCOT hourly, 5 yr | 240 h → 24–168 h | Adam (lr by grid search) | batch by grid search, **50 ep fixed** | MSE | RevIN, embedding E and MA kernel by grid search |
| TimesFM (R14/R2) | pretrain 100B pts; ERCOT | 512 (R2) | pretrain: cosine, peak 5e‑4 | – | – | 200M, patch in 32 / out 128, d 1280 |
| Chronos‑2 (R12/R3) | 54 energy datasets (R3) | 672–8000 → 40–1000 | zero‑shot; LoRA fine‑tune lr **1e‑6** | 1000 steps, batch 8 | quantile | 120M; LoRA r 8, α 16, on q,k,v,o + output patch embedding |
| TiRex‑2 (R13/R3) | as R3 | 672–8000 | zero‑shot | – | quantile | 38.4M active params, streams any context |
| RFKAN (R16) | PV + 6 weather vars, 5‑min | **12** → day‑ahead | not reported (pykan default LBFGS likely) | 50 steps × 6 grids = 300 | RMSE | grid 3→5→10→20→50→100, K 3, 12 input nodes |
| DT‑DANN (R18) | 4 Chinese PV plants, 15‑min | 96 → 96 | Adam (lr not reported) | not reported | MSE + CE | DANN λ = 2/(1+e^(−10p)) − 1; loss = L_pred,target + L_pred,source + L_domain |
| MTSA‑QRN (R8) | 40 MW PV plant, 15‑min | not reported | not reported | not reported | pinball | dilated TCN multi‑scale + MHSA + GAT; VMD input decomposition; adaptive CQR |
| RL selector (R25) | PV, 15‑min, 2017–18 | daily decisions | PPO actor‑critic | on‑policy rollout | reward = accuracy | 4 candidate models, 24‑dim state |

## 1. LSTM / GRU — the incumbent control

- **R20:**
  - **Model:** 2‑layer **bidirectional** LSTM with hidden size 256 (~2.6M params). Bidirectional means within the lookback window only, never across the horizon. Attention‑weighted pooling `c = Σ softmax(W h_t) h_t` instead of the last state.
  - **Training:** AdamW lr 1e‑3, weight decay 1e‑4, OneCycleLR; ≤200 epochs, early stop with patience 30 on validation MAPE; seed 42.
  - **Data:** chronological 70/15/15 split, lookback 240 h, horizons 24–168 h.
- **R2:** the final hidden state feeds a dense quantile head. Adam with early stopping; hyperparameters tuned through Nixtla's auto‑tuning, values not published. LSTM was the worst model at 24 h (nMAE 6.71%) and with 20% of the data (27.13%).
- **Baselines elsewhere:** R16 used a 3‑layer LSTM with Adam lr 1e‑3, batch 32. R32's CNN‑BiLSTM used Adam lr 1e‑3, batch 32, sequence 24, patience 10.
- **Suggestion:** use it as the floor every model must beat. Bidirectional encoding plus attention pooling is the stronger version to compare against.

## 2. PatchTST + ACI

- **R7 (PV, the closest to our task):**
  - **Data:** 5 kWp rooftop, hourly, 2014–2019. Trained on 4 years, tested on an unseen 12 months.
  - **Window:** 48 h lookback (two diurnal cycles), 1 h ahead.
  - **Inputs:** cyclical time encodings plus hourly meteorology.
  - **Settings:** batch 48, 10 epochs, dropout 0.1, d_model 128, 4 heads, patch 16, stride 8, `individual=True` (one head per channel). Optimizer and lr not reported.
  - **ACI:** applied after the model as post‑processing; α_t is kept within (0, 0.30) and γ is a small learning rate. The paper prints `α_{t+1} = α_t + γ(α_t − ε_t)`, which looks like a typo for the standard `α_t + γ(α − ε_t)`. Result: PICP 86.2% against a 90% target.
- **R20:** d_model 256, the shared AdamW/OneCycle recipe. Best on rhythmic, load‑only signals, but it gains about 3× less from weather covariates than iTransformer, because its channels are processed independently.
- **Original (F01):** lookback 336, patch 16, stride 8, 3 layers, 16 heads, D 128, FF 256, dropout 0.2, MSE. Longer lookback helps PatchTST more than other models.
- **Suggestion:** good without weather covariates. Use a 2–7 day lookback and patch 16 / stride 8. Add ACI after training rather than training probabilistically.

## 3. iTransformer

- **R20:** d_model 512 (shrinking to 256 keeps its weather advantage), same AdamW/OneCycle recipe. Weather enters as extra variable tokens, so the more covariates, the more it gains.
- **R6:** a PV comparator only. HDCF‑Mamba's MAE was up to 11.4% lower.
- **Original (F02):** Adam with lr in {1e‑3, 5e‑4, 1e‑4}, batch 32, a fixed 10 epochs, MSE, 2–4 blocks, D in {256, 512}, lookback 96.
- **Suggestion:** the right choice once real covariates exist (NWP, oracle weather). With only P/G/T and clear sky it has few variable tokens to attend over.

## 4. TimeXer

- **Original (F03):** Adam lr 1e‑4, MSE, 10 epochs with early stopping, 1–3 blocks, d_model in {128, 256, 512}, lookback 96. Patch 16 for long horizons and **24 for short‑term forecasts**. Exogenous series may have a different length and resolution than the target.
- **R2:** ~19M params, context 512. nMAE 3.85% at 24 h, 3.53% with weather covariates; unseen sites 6.81%, so it transfers worse than the foundation models.
- **Suggestion:** our future features (clear sky, calendar, yesterday's power) are exactly the exogenous inputs it is designed for. Try patch 24, lr 1e‑4, early stopping.

## 5. HDCF‑Mamba (the most complete PV recipe)

- **Data:** 4 PV datasets. ECD‑PV is desert, 15‑min, 12 months; the others are LSP‑PV, APS‑PV and PSB‑PV.
- **Inputs:** power plus module temperature, air temperature and humidity.
- **Preprocessing:**
  - per‑feature z‑score fitted on the training set;
  - removal of physically impossible values;
  - linear interpolation of gaps;
  - **moving‑average smoothing** before training.
- **Window:** horizons 24–512 steps; inputs up to 720 steps.
- **Training:**
  - **Adam lr 1e‑4**, β1 0.9, β2 0.999;
  - **ReduceLROnPlateau, factor 0.5**;
  - **early stopping with patience 4**;
  - MSE over the whole horizon, all steps predicted directly in one pass.
- **Hyperparameters:**
  - Optuna tuned d_model, batch size and the Mamba settings;
  - one setting was kept across all 4 datasets: d_state **32**, expand 2 (d_inner 256, so d_model 128), d_conv 4, dropout 0.1;
  - a single Mamba block (no stacking).
- **Suggestion:** our implementation defaults to d_state 16; the paper uses 32. Use the paper's recipe as written. Their low patience (4) relies on the plateau scheduler halving the lr first.

## 6. PowerMamba

- **R19:**
  - **Data:** ERCOT hourly (5 years, 22 series plus 262 with external forecasts) and PJM 2019–2024.
  - **Window:** context 240 h, horizons 24–168 h.
  - **Library:** Time‑Series‑Library.
  - **Training:** Adam with L2 loss, a **fixed 50 epochs**. Batch size per model, the embedding size E and the moving‑average kernel all chosen by grid search, values not published.
  - **Model:** RevIN rather than plain z‑score.
  - **External forecasts:** appended to the sequence (Sec. IV‑E); this cut renewable MSE from 0.590 to 0.142.
- **R20:** d_model 256, the shared AdamW/OneCycle recipe. It decomposes the input into trend and seasonal parts and encodes them bidirectionally.
- **Suggestion:** most valuable once a day‑ahead weather forecast exists. The external‑forecast path is the reason it wins.

## 7. TimesFM

- **R14 (pretraining):**
  - **Model:** decoder‑only, 200M params, input patch 32, output patch 128, d_model 1280.
  - **Training:** cosine decay with peak lr 5e‑4; real plus synthetic corpus (about 100B points).
- **R2:**
  - **Use:** pretrained checkpoint; context = 2 × horizon where possible, capped at the model's maximum (512 at 5‑min).
  - **Fine‑tuning:** done with "the same optimization pipeline" as the other models, values not reported.
  - **Results:** 24 h nMAE 3.71%; unseen sites 4.73%. With 20% of the data it reached 8.36%, against 27.13% for LSTM: the best data efficiency.
  - **Context length:** for solar, longer is better (4.28% at 6 h of context, 3.71% at 24 h).
- **R4:** TimesFM 2.5 adds covariates through a linear regressor fitted at inference over the context window.
- **Suggestion:** use it zero‑shot with the longest context we have (lookback 512+). It is the low‑data and new‑site candidate.

## 8. Chronos‑2

- **R12:** 120M params, zero‑shot. Group attention covers the target together with its past and known‑future covariates. Pretrained at context 2048, then extended to 8192.
- **R3 (54 energy datasets):**
  - **Settings:** contexts 672, 2000 and 8000; horizons 40–1000.
  - **Context:** performance **saturates at about 2000 steps** (about 3 weeks at 15‑min).
  - **Best mode:** zero‑shot with covariates; lowest median NRMSE 0.472, against XGBoost 0.611.
  - **LoRA fine‑tuning:** 1000 steps, **lr 1e‑6**, batch 8, 75% window overlap, rank 8, α 16, dropout 0, targets {q, k, v, o, output_patch_embedding}. It helped only the least predictable series. Its median got worse (0.599), and interval coverage collapsed on easy series.
- **R4 (cold‑start PV, 440 sites):** daily resolution, with a synthetic history from plant metadata plus ERA5 weather as context. Covariate‑aware foundation models were about 1.7–2× better than baselines; Chronos‑2 was the most robust when it had to feed its own forecasts back as context.
- **Suggestion:** use it zero‑shot with all covariates and a context of about 2–3 weeks. Skip LoRA fine‑tuning unless a site is very hard to predict.

## 9. TiRex‑2

- **R13:** recurrent (xLSTM‑based), 38.4M active parameters. Handles past and future covariates, stays stable when streamed over any context length, and costs the same per patch however long the history.
- **R3:** median NRMSE 0.474, practically tied with Chronos‑2. It ranked first on the leakage‑checked subset (0.36).
- **Our wrapper:** the installed `tirex-ts` package is TiRex 1 (univariate). TiRex‑2 needs its own release.
- **Suggestion:** zero‑shot, long context; it is the natural fit for a live digital‑twin stream.

## 10. RFKAN

- **R16:**
  - **Data:** PV power with temperature, humidity, cloud cover, irradiance, wind speed, and wind direction (binary‑coded); 5‑min data; min–max scaled to [0, 1]; chronological split.
  - **Test set:** **one day per season** (Mar 10, Jun 11, Sep 2, Dec 20), plus May 10 for the ablation.
  - **Window:** input seq_len **12** (1 h); the task is called day‑ahead.
  - **Grid schedule:** refined 3 → 5 → 10 → 20 → 50 → 100, with 50 steps per grid (300 iterations in total); grids above 100 overfit.
  - **Other settings:** K = 3; 12 input and 12 recurrent nodes.
  - **Optimizer:** not stated. The fit‑then‑refine schedule matches pykan, whose default is LBFGS.
- **Suggestion:** treat its gains as weak evidence (a tiny test set). To reproduce it, use a short input window (12–24 steps) and grid refinement; our implementation fixes the number of harmonics instead.

## 11. DEST‑GNN and the graph papers

R21 (DEST‑GNN), R22 (MSF‑GCN) and R23 (Fourier ST‑GNN) are paywalled (Elsevier) and were not read. What we know comes from the selection report: multi‑site, intra‑hour PV forecasting where each graph node is a site.

**Open question:** get these PDFs through the MIST library before tuning `graph_mamba` against them.

## 12. DT‑informed GAT/GCN + Mamba (SET Lab proposal)

There is no paper; it is a synthesis.
- **Temporal Mamba part:** inherit R6's recipe (Adam 1e‑4, plateau ×0.5, d_state 32, dropout 0.1).
- **Digital‑twin adaptation (optional):** inherit R18's DANN schedule.
- **Must beat:** the ablation chain Concat‑Mamba → GCN+Mamba → GAT+Mamba → DT‑GAT+Mamba (report §3).

## 13. DT Transformer + CNN + DANN

- **R18:**
  - **Data:** 2016–2018 PV competition data from 4 plants, 15‑min. Inputs: NWP‑**predicted** irradiance, wind speed and direction, temperature, humidity and pressure, plus measured irradiance and power. Min–max scaled.
  - **Domains:** one station is the "physical" target, with **1,400** training samples (2,800 in trial 2). The other three are the "digital" source, with 14,000 each.
  - **Window:** 96 → 96 (24 h → 24 h). Forecasts are generated recursively.
  - **Model:** CNN generator using conv, then max+avg pooling attention, batch norm, a skip connection and dropout; then a Transformer encoder‑decoder.
  - **Domain adaptation:** the classifier sits behind gradient reversal with λ = 2/(1+exp(−10p)) − 1.
  - **Loss:** MSE on the target + MSE on the source + cross‑entropy on the domain.
  - **Optimizer:** Adam; lr, batch size and epochs not reported.
  - **Results:** up to 39.99% better as the target set grows. With only 1,400 samples it lost to a plain Transformer in trial 1.
- **Suggestion:** use it for site transfer (NIST → SSP or the reverse) or season transfer, with the source labelled and the target lightly labelled. It needs a few thousand target samples before it pays off.

## 14. MTSA‑QRN + CQR

- **R8:**
  - **Data:** a 40 MW plant in northern China, SCADA at 15‑min, ultra‑short‑term.
  - **Inputs:** PV power history plus irradiance, temperature, humidity and wind speed.
  - **Preprocessing:** the power history is decomposed with **VMD** into intrinsic mode functions (number of modes chosen by an elbow criterion), and modes are ranked by six dependency measures.
  - **Training:** pinball loss, then adaptive CQR.
  - **Results:** CRPS 0.0400 before calibration; PICP90 = 0.9053 and CRPS 0.0453 after.
  - **Not reported:** optimizer, lr, batch size, epochs, split ratio, quantile levels.
- **Suggestion:** train with pinball loss on a quantile grid (our default 0.05–0.95, 7 levels), then calibrate with CQR on a held‑out calibration split. A VMD input decomposition is the one preprocessing step we don't have.

## 15. RL selector / fusion

- **R25 (PPO, the cleanest):**
  - **Data:** PV with day‑ahead weather station forecasts, 15‑min, 2017–2018; chronological split 0.64/0.16/0.20.
  - **Decision:** once a day, pick 1 of 4 **pretrained** models.
  - **State:** 24 dimensions: the meteorological context, the previous day's errors, and the previous action as a one‑hot.
  - **Training:** reward = the chosen model's accuracy; PPO with clipped objective and an on‑policy rollout buffer.
  - **Result:** RMSE 13.97% lower than the static‑optimal strategy.
- **R32:** CEEMDAN decomposition, then CNN‑BiLSTM per component (Adam lr 1e‑3, batch 32, sequence 24, patience 10), then a DQN that adds a correction. Actions are drawn from clustered training errors (ε‑greedy, replay buffer).
- **R33:** PPO with GAE plus dynamic copulas for daily probabilistic forecasts; 15‑min data, window of 1440 steps (15 days).
- **Suggestion:** this comes after the base models are trained. Compare against static‑best, inverse‑error weighting and a contextual bandit before using PPO.

## What this means for our training (SSP / NIST, 30‑min, ~10k training windows)

1. **Default recipe** (the R20 + R6 consensus): AdamW, lr 1e‑3 (attention/recurrent) or 1e‑4 (Mamba hybrids), weight decay 1e‑4; ReduceLROnPlateau ×0.5 or OneCycle; up to 100–200 epochs; early stopping on validation nMAE/nRMSE with patience 10–30; MSE loss; batch 32–64.
2. **Lookback:** at least 2 days (96 steps); 5–10 days for PatchTST and PowerMamba (R20 uses 240 h), and 2–3 weeks for the foundation models (R3 saturation).
3. **Scaling:** the papers z‑score or min–max scale using the training set; ours is physical and fit‑free, which is fine. Smoothing (R6) and VMD/CEEMDAN decomposition (R8, R32) are optional data steps worth one experiment each.
4. **Hyperparameter search is how they trained:** Optuna (R6), grid search (R19), Nixtla (R2). Plan a small search per model rather than trusting defaults.
5. **Controls the framework needs** (added to `TrainingRecipe`):
   - an early‑stopping metric other than the loss (validation nMAE or nRMSE);
   - a fixed‑epoch mode with no early stopping (R19, F02, F03);
   - learning‑rate warmup;
   - a training‑data fraction for data‑efficiency curves (R2);
   - initialising from a trained run, optionally with frozen parts, for transfer or fine‑tuning (R2 unseen sites, R18);
   - a small hyperparameter search.
6. **Paper recipes as presets:** `pvf.paper_recipe("<model>")` returns the hyperparameters and training recipe each paper used, filled in where the paper is silent with the defaults from point 1.
