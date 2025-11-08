## RealMLP Best Practices, DOs and DON'Ts

Date: 2025-08-19

### Scope
Practical, production-aligned guidance for integrating a RealMLP-style tabular neural network into StockPredictor. Focus areas: preprocessing, categorical handling, architecture, training, confidence/thresholding, evaluation, MLflow, prediction/export, and operations.

### Data & Preprocessing
- DO: Use robust normalization for numerics
  - Smooth clipping to quantiles q1=0.01 and q2=0.99 (per-feature) then RobustScaler.
  - Persist clipping stats and scaler; validate at inference; warn on distribution drift.
- DO: Keep single, consistent scaler across tuning/training/evaluation/prediction to avoid leakage.
- DO: Consider return-aware target capping (e.g., clamp target residual outliers to a symmetric band like [-0.3, 0.3]) after quantile clamp.
- DO: Treat `date_int` as numeric (time is ordinal/metric). Optionally add cyclical encodings (day-of-week/month sin/cos) without creating leakage.
- DON'T: Feed raw NaN/Inf; sanitize early and log counts using the project logger.

### Categorical Handling (ticker_id)
- DO: Use `ticker_id` as an embedding input via an index column `ticker_id_idx` for the model.
  - Persist the `ticker_id`→`ticker_id_idx` mapping in MLflow artifacts.
  - Handle unseen IDs by routing to an OOV index (0) with explicit logging.
- DO: Keep raw `ticker_id` in the DataFrame as metadata for evaluation/export alignment.
- DON'T: Feed raw integer `ticker_id` as a numeric feature (it imposes fake ordinal structure and invites spurious correlations).
- DO: Regularize embeddings
  - Embedding dim: cap at ≤64 using dim = min(64, round(1.6 * n_unique**0.56)).
  - Embedding dropout: 0.05–0.1.
  - L2 weight decay on embedding parameters.

### Features
- DO: Maintain a “required columns” set for evaluation/export alignment: `close`, `date_int`, `ticker_id`.
- DO: Keep model input features separate from meta columns (split `X_model` vs `X_meta`).
- DON'T: Introduce features that leak future information.

### Architecture
- DO: Use the RealMLP trunk
  - Optional NumericEmbedding → DiagonalFeatureScaler → [Linear → BN → Activation → Dropout] × L → Output.
  - Preferred activation: GELU (optionally SiLU/Swish or PReLU for A/B tests).
  - Weight init: Kaiming for ReLU/GELU, BN weight=1, bias=0; biases zeros.
- DO: Place DiagonalFeatureScaler after numeric normalization and before the first trunk block.
- DO: Keep hidden sizes in a moderate-to-large path (e.g., [512, 256, 128, 64]) and BN enabled.
- DO: Expose LayerNorm as a fallback for small or unstable batch regimes.
- Optional: Gradient checkpointing for deeper networks to reduce VRAM.

### Training
- DO: Optimizer & schedules
  - AdamW with weight_decay≈1e-5; LR≈1e-3 with CosineAnnealingLR; optionally OneCycleLR.
- DO: Loss: consider Huber loss (delta≈0.05–0.1) for robustness against residual outliers; keep MSE as a toggle.
- DO: Mixed precision on by default for 16GB VRAM; monitor loss scaling/NaNs.
- DO: DataLoader: zero-copy `torch.from_numpy` on float32 arrays; `num_workers` tuned to CPU cores; `pin_memory=True` only if CUDA; `persistent_workers=True` when workers>0.
- Optional: memmap-backed dataset for >1M-row flows to limit RAM.

### Confidence & Thresholding
- DO: Default confidence method for MLP: magnitude-based (variance/simple/margin), normalized to [0,1].
- DO: Add `mc_dropout` confidence (T≈20 stochastic passes) for improved epistemic uncertainty when feasible.
- DO: Keep threshold optimization centralized via `ThresholdEvaluator` + `ThresholdPolicy`.
- Optional: Extend `ThresholdPolicy` with per-group strategies (`group_key='date_int'`), e.g., top-k per day or per-date quantiles, when diversification is required.

### Evaluation & Validation
- DO: Use financial-time-series-safe validation: PurgedKFold with embargo to avoid leakage across overlapping targets.
- DO: Maintain rolling/blocked backtest windows (e.g., quarterly/yearly) and report per-block metrics to detect regime drift.
- DO: Track both traditional metrics (MSE/MAE/R²) and profit-aware metrics (total profit, profit per investment, investment success rate, custom accuracy).
- DO: Calibrate confidence post-hoc (e.g., isotonic regression on a calibration split) if threshold stability is critical.

### MLflow & Artifacts
- DO: Log model and artifacts comprehensively
  - `model/` via mlflow.pytorch.
  - Artifacts: `preprocessor/robust_scaler.pkl`, `preprocessor/clip_stats.json`, `preprocessor/feature_names.json`, `preprocessor/cat_maps.json` (for `ticker_id` mapping).
  - Params/Tags: architecture flags (diagonal/numeric embedding on/off), embedding dim/dropout, loss type, scheduler, gradient clipping, mixed precision, feature schema hash.
- DO: Add a small `X_eval` input example; convert integer columns to float64 for signature.

### Prediction & Export
- DO: Reuse `BasePredictor.save_predictions_to_excel`; set `model_type='realmlp'` so files save under `predictions/realmlp/`.
- DO: File naming: `realmlp_YYYYMMDDHHMM.xlsx`.
- DO: Preserve the project’s standard schema (includes `ticker_id`, `date_int`, `date`, `ticker`, `company_name`, `predicted_return`, `predicted_price`, `current_price`, `actual_return`).
- DO: Keep meta columns (`close`, `date_int`, `ticker_id`) in the features DataFrame for alignment but do not feed raw `ticker_id` into the model.

### Performance & Scaling
- DO: Batch sizes 512–2048 depending on VRAM; ensure headroom for embedding tables.
- DO: Enable mixed precision; consider gradient checkpointing for deeper stacks.
- DO: Profile GPU/VRAM usage and log per-epoch LR; capture training curves to MLflow.

### Benchmarking & Ablations
- DO: Benchmark RealMLP against current MLP and best LightGBM/XGBoost baselines on identical folds.
- DO: Run quick ablations: diagonal on/off, embeddings on/off, GELU vs SiLU, Huber vs MSE, Cosine vs OneCycle, MC-dropout vs magnitude confidence.

### MLOps & Production
- DO: Use the centralized project logger (no print) for all modules.
- DO: Keep Windows+uv environment guidelines (uv run, one command per line) for scripts/README updates.
- DO: Monitor drift: periodic snapshots of feature distributions and embedding index coverage; alert if unseen `ticker_id` rate spikes.

### Common Pitfalls (DON'Ts)
- DON'T: Treat `ticker_id` as a numeric feature; always embed or exclude from model inputs.
- DON'T: Fit scaler on validation/test/prediction data; fit only on train data and reuse.
- DON'T: Use group-wise statistics or feature engineering that can leak target information across time.
- DON'T: Run confidence methods requiring dropout (e.g., MC-dropout) without enabling dropout at inference when using them.
- DON'T: Change feature order; always align to persisted `feature_names` and validate strictly at load/predict time.

### Quick Checklist
- Preprocessing: q1/q2 clip + RobustScaler; artifacts saved.
- Categorical: `ticker_id_idx` for model; raw `ticker_id` kept as metadata; OOV handled.
- Architecture: diagonal scaler; GELU; BN; proper init.
- Training: AdamW; Cosine/OneCycle; Huber optional; mixed precision on.
- Confidence/threshold: default magnitude; optional MC-dropout; centralized policy + evaluator.
- Evaluation: purged k-fold; profit metrics; calibration optional.
- MLflow: model + preprocessor artifacts + params/tags + schema hash.
- Export: reuse BasePredictor; `predictions/realmlp/realmlp_YYYYMMDDHHMM.xlsx`.



### Additional Recommendations
- Data-dependent initialization: optionally run LSUV/data-dependent init after Kaiming to stabilize early training.
- Conformal prediction: add split-conformal intervals on returns to complement confidence for decision robustness.
- Group-aware thresholds (per ticker): extend `ThresholdPolicy` with `group_key='ticker_id'` (e.g., top-k/quantile per ticker) for diversification controls.
- Export metadata sheet: include model version, data window, threshold, confidence method, and key params in an extra Excel sheet.