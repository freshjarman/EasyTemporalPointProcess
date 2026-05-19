# EasyTemporalPointProcess (EasyTPP) Guidebook

> A complete walk‑through of the `EasyTPP` repository — both for understanding the framework as an MTPP practitioner and for using it as the launching pad of your own Marked Temporal Point Process (MTPP) research.
>
> Scope: code as currently checked into this repo (PyTorch‑only after 2025‑11‑06, including the WSMTHP and S2P2 additions).
>
> Audience: researchers / engineers who already know the basics of point processes and PyTorch.

---

## Table of contents

1. [What is in this repository](#1-what-is-in-this-repository)
2. [MTPP background you must keep in mind](#2-mtpp-background-you-must-keep-in-mind)
3. [Top‑level architecture and execution pipeline](#3-top-level-architecture-and-execution-pipeline)
4. [Configuration system (`config_factory`)](#4-configuration-system-config_factory)
5. [Data pipeline (`preprocess`)](#5-data-pipeline-preprocess)
6. [Runners and model wrapper](#6-runners-and-model-wrapper)
7. [The model base class — what every TPP must implement](#7-the-model-base-class--what-every-tpp-must-implement)
8. [Thinning algorithm for prediction / generation](#8-thinning-algorithm-for-prediction--generation)
9. [Reusable layers (`torch_baselayer`)](#9-reusable-layers-torch_baselayer)
10. [State‑Space submodule (`easy_tpp/ssm`)](#10-state-space-submodule-easy_tppssm)
11. [Implemented models — line‑by‑line guide](#11-implemented-models--line-by-line-guide)
    1. [RMTPP (KDD'16)](#111-rmtpp-kdd16)
    2. [NHP (NeurIPS'17)](#112-nhp-neurips17)
    3. [FullyNN (NeurIPS'19)](#113-fullynn-neurips19)
    4. [SAHP (ICML'20)](#114-sahp-icml20)
    5. [THP (ICML'20)](#115-thp-icml20)
    6. [IntensityFree (ICLR'20)](#116-intensityfree-iclr20)
    7. [ODETPP (ICLR'21 simplified)](#117-odetpp-iclr21-simplified)
    8. [AttNHP (ICLR'22)](#118-attnhp-iclr22)
    9. [ANHN (IJCNN'21, optional)](#119-anhn-ijcnn21-optional)
    10. [WSMTHP (NeurIPS'24)](#1110-wsmthp-neurips24)
    11. [S2P2 (NeurIPS'25)](#1111-s2p2-neurips25)
12. [Hyper‑parameter optimisation (`hpo`)](#12-hyper-parameter-optimisation-hpo)
13. [Metrics & evaluation conventions](#13-metrics--evaluation-conventions)
14. [End‑to‑end usage walkthroughs](#14-end-to-end-usage-walkthroughs)
15. [Cheat‑sheet for adding a new model](#15-cheat-sheet-for-adding-a-new-model)
16. [Cheat‑sheet for adding a new dataset](#16-cheat-sheet-for-adding-a-new-dataset)
17. [Known pitfalls / common bugs](#17-known-pitfalls--common-bugs)
18. [Glossary of tensor shape symbols](#18-glossary-of-tensor-shape-symbols)

---

## 1. What is in this repository

`EasyTPP` is an open‑source PyTorch toolkit for benchmarking and developing **neural Marked Temporal Point Processes** (MTPPs). It was published at ICLR 2024 (Xue et al.) and has since absorbed several follow‑up papers (e.g. WSMTHP NeurIPS'24, S2P2 NeurIPS'25). Everything is structured around three pillars:

```
                 +------------- YAML config -------------+
                 |                                        |
                 v                                        v
        +-----------------+                     +------------------+
        |  DataConfig /   |                     |  ModelConfig /   |
        |  TrainerConfig  |                     |  ThinningConfig  |
        +-----------------+                     +------------------+
                 |                                        |
                 v                                        v
    +-------------------------+              +--------------------------+
    |  TPPDataLoader          |   batches    |  TorchBaseModel sub-     |
    |  (HuggingFace datasets, |  ---------->|  class (NHP / THP / ...) |
    |  pkl loader, tokenizer, |              |  + EventSampler (thin.)  |
    |  collator, padding)     |              +--------------------------+
    +-------------------------+                        ^
                 |                                      |
                 +----------------- TPPRunner ----------+
                                    |
                                    v
                       checkpoints / logs / metrics
```

Folder map (top of repo):

```
easy_tpp/
├── __init__.py              # exposes version
├── config_factory/          # All *Config classes + yaml ↔ object
│   ├── config.py            # Base Config (Registrable)
│   ├── data_config.py
│   ├── model_config.py      # TrainerConfig / ThinningConfig / BaseConfig / ModelConfig
│   ├── runner_config.py     # RunnerConfig wires the four above together
│   └── hpo_config.py
├── preprocess/              # Dataset / tokenizer / collator / loader
│   ├── dataset.py
│   ├── event_tokenizer.py
│   ├── data_collator.py
│   └── data_loader.py       # TPPDataLoader (json/pkl/HuggingFace)
├── model/
│   ├── __init__.py          # registers every Torch* model
│   └── torch_model/
│       ├── torch_basemodel.py   # ABC for every TPP, with logL helpers
│       ├── torch_baselayer.py   # MultiHeadAttention, ScaledSoftplus, DNN, ...
│       ├── torch_thinning.py    # EventSampler — thinning algorithm
│       ├── torch_nhp.py         # NHP (continuous‑time LSTM cell)
│       ├── torch_rmtpp.py
│       ├── torch_thp.py
│       ├── torch_sahp.py
│       ├── torch_attnhp.py
│       ├── torch_anhn.py
│       ├── torch_fullynn.py
│       ├── torch_intensity_free.py
│       ├── torch_ode_tpp.py
│       ├── torch_s2p2.py        # NeurIPS'25
│       └── torch_wsm_thp.py     # NeurIPS'24
├── ssm/                     # Deep continuous‑time SSM layers used by S2P2
├── runner/
│   ├── base_runner.py       # Runner ABC + Registrable
│   └── tpp_runner.py        # std_tpp runner (train / eval / gen)
├── hpo/                     # Optuna integration
├── default_registers/       # Metric registration (rmse, acc)
├── torch_wrapper.py         # TorchModelWrapper (optimiser, tensorboard, ...)
└── utils/                   # logging, metrics helper, registrable, ode_utils, ...

examples/                    # Ready‑to‑run training scripts + configs
notebooks/                   # Tutorials
docs/                        # Sphinx documentation + this guidebook
tests/                       # Unit tests
```

---

## 2. MTPP background you must keep in mind

A Marked Temporal Point Process is a stochastic sequence of events `(t_i, k_i)` with `t_1 < t_2 < ...` (timestamps) and `k_i ∈ {1, ..., K}` (mark / event type). Given the past `H_t = {(t_j, k_j) : t_j < t}`, an MTPP is fully specified by the **conditional intensity** of each mark:

```
λ*_k(t) = lim_{Δt→0} P(event of type k in [t, t+Δt) | H_t) / Δt
```

For an observed sequence on `[0, T]`, the (per‑mark) log‑likelihood is

```
log L = Σ_i log λ*_{k_i}(t_i)  −  Σ_k ∫_0^T λ*_k(s) ds
       (event terms)            (survival/non-event term)
```

Most papers in this repo follow that recipe directly. The integral usually has no closed form and is approximated by:

- **Monte‑Carlo on the inter‑event interval** (`use_mc_samples=True` in `TorchBaseModel.compute_loglikelihood`).
- **Trapezoidal rule** on a regular grid (the alternative branch in the same function).

Some methods sidestep the intensity entirely:
- **IntensityFree** parameterises `p(τ | H)` (a log‑normal mixture) directly.
- **FullyNN** models the cumulative hazard with a monotone net and obtains intensity by automatic differentiation.
- **WSMTHP** does not maximise log‑likelihood at all during training; it uses **Weighted Score Matching** on the total intensity.

The terminology you will meet everywhere in the codebase:

| Variable      | Meaning                                             | Shape                       |
|---------------|-----------------------------------------------------|-----------------------------|
| `time_seqs`   | Absolute event times `t_i`                          | `[B, N]`                    |
| `time_delta_seqs` | `τ_i = t_i − t_{i-1}` (with `τ_0 = 0`)          | `[B, N]`                    |
| `type_seqs`   | Mark ids; padded with `pad_token_id`                | `[B, N]` (long)             |
| `batch_non_pad_mask` | `True` where the event is real                | `[B, N]` (bool)             |
| `attention_mask` | Combined causal + key‑pad mask (`1`=mask, `0`=keep) | `[B, N, N]` (uint8)     |

---

## 3. Top‑level architecture and execution pipeline

The canonical flow when you run `python train_nhp.py` is:

```
yaml file ──► Config.build_from_yaml_file ──► RunnerConfig
                                                  │
RunnerConfig ──► Runner.build_from_config ──► TPPRunner('std_tpp')
                                                  │
   TPPRunner.__init__:
     • TPPDataLoader (train / valid / test)       ◄── statistics like mean/std of log‑dt fed back to IntensityFree
     • TorchBaseModel.generate_model_from_config  ◄── picks subclass by model_id
     • TorchModelWrapper(model, base, model, trainer)   ◄── optimiser + tensorboard
                                                  │
   runner.run() dispatches on `stage`:
     • train  ──► _train_model → run_one_epoch(TRAIN | VALIDATE)
     • eval   ──► _evaluate_model
     • gen    ──► _gen_model → multi‑step prediction → pred.pkl
```

Key insight: a single yaml file can contain **many experiments**. Each top‑level key (e.g. `NHP_train`, `S2P2_train`) is an *experiment_id*; the `data:` block holds dataset definitions referenced by `base_config.dataset_id`. The CLI just selects which experiment to run.

---

## 4. Configuration system (`config_factory`)

All config classes derive from `Config` (abstract) which is also `Registrable`. The class is selected by the yaml key `pipeline_config_id`:

- `runner_config`  →  `RunnerConfig` (training/eval/gen pipeline)
- `hpo_runner_config`  →  `HPORunnerConfig` (HPO pipeline)

A `RunnerConfig` is composed of four sub‑configs (`easy_tpp/config_factory/runner_config.py`):

```
RunnerConfig
├── data_config     (DataConfig)
│   ├── train_dir, valid_dir, test_dir, data_format (json|pkl)
│   └── data_specs  (DataSpecConfig)
│        ├── num_event_types, pad_token_id
│        ├── padding_side / padding_strategy / max_len
│        └── truncation_side / truncation_strategy
├── base_config     (BaseConfig)
│   ├── stage (train|eval|gen)
│   ├── backend ('torch' — only option after 2025‑11‑06)
│   ├── dataset_id, runner_id ('std_tpp'), model_id
│   ├── exp_id, base_dir
│   └── specs (filled at runtime: log_folder, saved_model_dir, ...)
├── trainer_config  (TrainerConfig)
│   ├── seed, gpu, batch_size, max_epoch, shuffle
│   ├── optimizer, learning_rate, valid_freq, use_tfb
│   └── metrics (['acc','rmse'])
└── model_config    (ModelConfig)
    ├── hidden_size, time_emb_size, num_layers, num_heads
    ├── dropout_rate, use_ln, sharing_param_layer
    ├── loss_integral_num_sample_per_step, use_mc_samples
    ├── pretrained_model_dir (set to restore a checkpoint)
    ├── thinning  (ThinningConfig — see §8)
    └── model_specs  (free‑form dict, where model‑specific knobs live)
```

`RunnerConfig.update_config()` does the wiring:

1. Creates `checkpoints/<unique_id>/` containing `models/saved_model`, the dumped `*_output.yaml`, optional `tfb_*` dirs.
2. Pushes `num_event_types`, `num_event_types_pad`, `pad_token_id`, `max_len` from `data_specs` into `model_config`.
3. Sets `model_config.is_training` based on `stage`.
4. **Special path for `WSMTHP`** (`_maybe_set_max_observed_time`): depending on `T_mode` (`manual` / `batch` / `train_global`) it pre‑computes the observation window end `T` by scanning the training split, because WSM needs it as a constant.

> Take‑away: when you create a brand‑new model that needs an unusual config field, put it under `model_specs:` in yaml and read it via `model_config.model_specs.get('your_field', default)`.

---

## 5. Data pipeline (`preprocess`)

### 5.1 Expected dataset format

Two on‑disk formats are supported:

**Pickle (Gatech style)** — `data['dim_process']` plus splits `train/dev/test`, each a list of sequences; each sequence is a list of dicts:

```python
{"time_since_start": float,
 "time_since_last_event": float,
 "type_event": int}
```

**JSON (HuggingFace datasets)** — three columns: `time_since_start`, `time_since_last_event`, `type_event`, plus the metadata column `dim_process`. Either provide a local `.json` or a HuggingFace repo prefixed with `easytpp/` (the toolkit auto‑downloads via `datasets.load_dataset`).

### 5.2 `TPPDataset`

`easy_tpp/preprocess/dataset.py` wraps the lists into a `torch.utils.data.Dataset`. The crucial helper `get_dt_stats()` returns `(mean(log dt), std(log dt), min dt, max dt)` — these statistics are pushed into `IntensityFree`'s log‑normal AffineTransform automatically by `Runner.__init__`.

### 5.3 `EventTokenizer` + `TPPDataCollator`

`EventTokenizer` is heavily inspired by HuggingFace's `PreTrainedTokenizer`. It defines a fixed `model_input_names`:

```python
['time_seqs', 'time_delta_seqs', 'type_seqs', 'batch_non_pad_mask', 'attention_mask']
```

`pad()` then `_pad()`:
- Resolve padding strategy (`longest`, `max_length`, `do_not_pad`).
- Right‑ or left‑pad event sequences with `pad_token_id` (= `num_event_types`).
- Build `batch_non_pad_mask` (True where event is real) and `attention_mask` (combined causal + key‑padding, `True` = mask, `False` = keep).
- Convert to numpy then to PyTorch via `BatchEncoding(... tensor_type='pt')`.

The `TPPDataCollator` is just a thin dataclass that forwards a `list[dict]` from the `DataLoader` to `tokenizer.pad(...)`.

### 5.4 `TPPDataLoader`

`easy_tpp/preprocess/data_loader.py` is the high‑level facade:

- `train_loader()`, `valid_loader()`, `test_loader()` — return `torch.utils.data.DataLoader` instances wrapped around `TPPDataset` with the collator above.
- `build_input(...)` dispatches on `data_format`. For `json` it can read a local file *or* a HuggingFace dataset.
- `get_max_event_time(split)` / `get_global_max_event_time()` — used by WSMTHP and similar window‑aware models.
- `get_statistics(split)` and `plot_*` methods provide quick EDA.

> When a `DataLoader` yields a batch, what your model receives in `forward` is a 5‑tuple `(time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask)` of `torch.Tensor`s. This convention is universal across **all** implemented models.

---

## 6. Runners and model wrapper

### 6.1 `Runner` (abstract)

`easy_tpp/runner/base_runner.py` defines the registrable contract. Subclasses must implement `_train_model`, `_evaluate_model`, `_gen_model`, `_save_model`, `_load_model`. The base class:

- Instantiates `TPPDataLoader` once.
- **Calls `train_loader.dataset.get_dt_stats()` and writes `mean_log_inter_time / std_log_inter_time` into `model_config`** (needed by `IntensityFree`).
- Provides public `train`, `evaluate`, `gen`, `run` methods.

### 6.2 `TPPRunner` (the standard)

Registered as `std_tpp` (`@Runner.register(name='std_tpp')`).

```
_train_model(train_loader, valid_loader, test_loader):
  for epoch in range(max_epoch):
      train_metrics = run_one_epoch(train_loader, TRAIN)
      if epoch % valid_freq == 0:
          valid_metrics = run_one_epoch(valid_loader, VALIDATE)
          if valid loglike is the best so far  ──►  save checkpoint

_evaluate_model(loader):
  run_one_epoch(loader, VALIDATE), log final metrics

_gen_model(loader):
  run_one_epoch(loader, PREDICT) → pickle to pred.pkl
```

The `run_one_epoch` routine accumulates loss, builds `epoch_pred`, `epoch_label`, `epoch_mask`, then computes registered metrics (`rmse`, `acc`) at the end of the epoch.

### 6.3 `TorchModelWrapper`

`easy_tpp/torch_wrapper.py` adapts the model for the wrapper:

- Moves model to the chosen device.
- Builds the optimiser (`adam` is the default).
- Owns TensorBoard `SummaryWriter`s.
- `save() / restore()` thin wrappers for `state_dict`.
- `run_batch(batch, phase)`:
  - **TRAIN**: forward, `loss/num_event` → `backward()` → `optimizer.step()`.
  - **VALIDATE**: forward to get loss; if the model has an `event_sampler`, switch to `eval()` and call `predict_one_step_at_every_event` for next‑event prediction (used for `rmse` / `acc`).
  - **PREDICT**: call `predict_multi_step_since_last_event` — multi‑step generation since the last observed event.

Special‑case: `FullyNN` needs `requires_grad=True` even at validation (it uses `autograd.grad` to recover intensity).

---

## 7. The model base class — what every TPP must implement

`easy_tpp/model/torch_model/torch_basemodel.py` is the single most important file in the project. Every model **must** subclass `TorchBaseModel` and provide at least:

```python
class MyModel(TorchBaseModel):
    def __init__(self, model_config): ...
    def loglike_loss(self, batch) -> (loss, num_events): ...
    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs,
                                            type_seqs, sample_dtimes, **kwargs): ...
    # optional, default implementations are provided
    def predict_one_step_at_every_event(self, batch): ...
    def predict_multi_step_since_last_event(self, batch, forward=False): ...
```

`TorchBaseModel.__init__` does the housekeeping:

- Stores `hidden_size`, `num_event_types` (without PAD/BOS/EOS), `num_event_types_pad` (with PAD), `pad_token_id`, `loss_integral_num_sample_per_step`, `use_mc_samples`, `gen_config` (thinning config), and device.
- Creates a default `layer_type_emb = nn.Embedding(num_event_types_pad, hidden_size, padding_idx=pad_token_id)`. Models that need a different embedding (e.g. `S2P2`) override it to `None`.
- If `thinning:` is set in yaml, instantiates `EventSampler` (see §8) and stores it in `self.event_sampler`.

### 7.1 The two shared helpers

- `make_dtime_loss_samples(time_delta_seq)` — generates `loss_integral_num_sample_per_step` equally spaced points in each interval `[0, τ_i]`. Output shape: `[B, N, G]`. Used to estimate ∫ λ ds.
- `compute_loglikelihood(time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask, type_seq)` — the universal log‑likelihood:
  1. **Event term** — `nll_loss` over the marked intensity at the actual event mark, ignoring `pad_token_id`. Equivalent to `Σ_i log λ*_{k_i}(t_i)`.
  2. **Non‑event term** — `(Σ_m λ_m)(t_i + τ ratios)`, then either MC mean or trapezoidal rule, multiplied by `τ_i * seq_mask`. Equivalent to `∫_{t_{i-1}}^{t_i} λ_tot(s) ds`.

Returns `(event_ll, non_event_ll, num_events)`; subclasses combine them as `loss = -(event_ll - non_event_ll).sum()`.

### 7.2 The two shared predictors

`predict_one_step_at_every_event(batch)` — the standard "one step ahead at every event in the sequence" evaluation used for `rmse` / `acc`:
1. Drop the last event (no label after it).
2. Compute an intensity upper bound and draw candidate `τ` via the thinning sampler (`EventSampler.draw_next_time_one_step`).
3. The expected next time is `Σ τ_sample * weights`.
4. The next mark is `argmax_m Σ_s w_s · λ_m(t + τ_s) / Σ_m λ_m(t + τ_s)`.

`predict_multi_step_since_last_event(batch, forward=False)` — autoregressive multi‑step rollout:
- Trim the last `num_step_gen` events.
- For `i in range(num_step_gen)`: thinning to get `τ` and mark; append the predicted event to the sequence and continue.

Both are completely model‑agnostic: every subclass only has to implement `compute_intensities_at_sample_times` and the base provides predict + train scaffolding for free.

---

## 8. Thinning algorithm for prediction / generation

`easy_tpp/model/torch_model/torch_thinning.py` implements **Ogata's modified thinning** algorithm:

1. Sample `num_samples_boundary` points uniformly inside `[0, τ]`, evaluate the *total* intensity at each, take `over_sample_rate × max` as the upper bound `λ*`.
2. Draw `num_exp` exponential variables with rate `λ*`; `cumsum` gives candidate inter‑event offsets.
3. Evaluate the actual total intensity at those offsets.
4. Draw uniform `u ~ U(0,1)`; accept the first candidate where `u · λ* < Σ_m λ_m`.
5. Mark prediction is the **mark with the highest posterior at the accepted time**.

Knobs (under `thinning:` in yaml, mapped to `ThinningConfig`):

| Field                  | Meaning                                                                      |
|------------------------|------------------------------------------------------------------------------|
| `num_sample`           | Number of i.i.d. predictions used to average dtime and weight marks.          |
| `num_exp`              | Number of exponential candidates per prediction.                              |
| `over_sample_rate`     | Safety factor on the intensity upper bound.                                   |
| `num_samples_boundary` | Grid used to estimate the upper bound.                                        |
| `dtime_max`            | Hard cap on the predicted dtime if all candidates are rejected.               |
| `patience_counter`     | Reserved (adaptive thinning).                                                 |
| `num_step_gen`         | How many future events to roll out in `predict_multi_step_since_last_event`.  |

> All models that expose an intensity function inherit thinning for free. Intensity‑free models (`IntensityFree`) override `predict_one_step_at_every_event` to sample directly from their parametric distribution.

---

## 9. Reusable layers (`torch_baselayer`)

`easy_tpp/model/torch_model/torch_baselayer.py` collects the building blocks that several models share:

| Layer                          | Purpose                                                                  |
|--------------------------------|--------------------------------------------------------------------------|
| `ScaledSoftplus`               | Per‑mark learnable softplus `β_k` (Eq. 4a of NHP; used everywhere).      |
| `MultiHeadAttention`           | Standard MHA but with **mask convention `1 = mask`** (note!).            |
| `EncoderLayer`                 | Wraps MHA (+ optional FFN). Used by THP/SAHP/AttNHP/WSMTHP encoders.     |
| `TimePositionalEncoding`       | Sinusoidal absolute‑time encoding (THP, Eq. 2).                           |
| `TimeShiftedPositionalEncoding`| Time‑aware sinusoid + learned phase shift (SAHP, Eq. 8).                  |
| `DNN`                          | Plain MLP with configurable activation / dropout / batchnorm.            |
| `rk4_step_method` (in `utils/ode_utils.py`) | Fixed‑step Runge‑Kutta 4 used by `ODETPP`.                  |

`ScaledSoftplus` is worth memorising — it is the activation that converts the model's pre‑intensity into a non‑negative per‑mark intensity. Its `β` is learnable per mark, which empirically helps when mark frequencies are heterogeneous.

---

## 10. State‑Space submodule (`easy_tpp/ssm`)

This sub‑package houses **S5‑style deep continuous‑time linear state‑space layers**, used exclusively by `S2P2` (NeurIPS'25). Files:

- `initializers.py` — DPLR‑HiPPO initialisation of the diagonal state matrix `Λ`.
- `ssm_util.py` — `discretize_zoh`, `apply_ssm` (parallel scan or for‑loop recurrence).
- `models.py` — three layer variants:
  - `LLH` (Latent Linear Hawkes layer): impulses at event times, intensity defined on the right limit only.
  - `Int_Forward_LLH`: integrates `Bu_t` with a ZOH on `u_t` *forward in time*.
  - `Int_Backward_LLH`: same but ZOH on `u_{t'-}` *backward*, which the paper recommends and is the default in the example config.

Each layer exposes:
- `forward(left_u_NH, right_u_NH, mark_embedding, dt, initial_state)` → returns right limits of `x` and `u` at every event;
- `get_left_limit(right_limit, dt, current_right_u, next_left_u)` → evolves the latent for arbitrary `dt`;
- `depth_pass(left_x, left_u, prev_right_u)` → applies `C`, `D`, residual, activation, layer‑norm — the "depth" axis of the network.

If you ever want to read just one model to understand the abstractions, read `LLH.forward` together with `S2P2.forward` side by side.

---

## 11. Implemented models — line‑by‑line guide

Below each model has the same template: **paper → key equations → architecture → how the code maps the equations → loss → prediction → relevant `model_specs`**.

### 11.1 RMTPP (KDD'16)

- **Paper**: *Recurrent Marked Temporal Point Processes*, Du et al., 2016.
- **Idea**: Plain RNN over `(type_emb + time_emb)`. Intensity is exponential in time since last event with mark‑specific scale.

```
h_i = RNN([type_emb(k_i) + linear(t_i)])
λ*_m(t) = exp( linear_m(h_i) + w_m · (t − t_i) + b_m )            (Eq. 11)
```

- **Code**: `torch_rmtpp.py`.
  - `evolve_and_get_intentsity` implements the closed‑form clipped exp.
  - `forward` returns left‑limit intensities at `t_1..t_N` and right‑limit hidden states (needed for sampling continuation).
  - `loglike_loss` uses `make_dtime_loss_samples` + `compute_loglikelihood`. The integral has a closed form too but is computed by MC for code uniformity.

- **Useful config**: `time_emb_size`, `mc_num_sample_per_step`, all under `experiment_config.yaml::RMTPP_train`.

### 11.2 NHP (NeurIPS'17)

- **Paper**: *The Neural Hawkes Process*, Mei & Eisner, 2017.
- **Idea**: A **continuous‑time LSTM** whose cell state decays exponentially between events:

```
c(t) = c̄_i + (c_i − c̄_i) · exp(−δ_i · (t − t_i))                 (Eq. 7)
h(t) = o_i ⊙ tanh(c(t))
λ*(t) = ScaledSoftplus( W · h(t) )                                 (Eq. 4a)
```

- **Code**: `torch_nhp.py`.
  - `ContTimeLSTMCell` is the parameterised cell. `recurrence(event_emb, h_t-, c_t-, c̄_t-1)` returns `(c_i, c̄_i, δ_i, o_i)`. `decay(c, c̄, δ, o, dt)` evolves them forward by `dt`.
  - `forward` unrolls the cell event by event, storing **left** hidden states for `t_1..t_N` (just after decay) and **right** states for `t_0..t_N` (just after the update). The "right at `t_N`" is what `compute_intensities_at_sample_times` decays from when sampling the future.

- **Useful config**: `hidden_size`, plus optional `model_specs.beta`, `model_specs.bias`.

### 11.3 FullyNN (NeurIPS'19)

- **Paper**: *Fully Neural Network based Model for General Temporal Point Processes*, Omi, Aihara & Ueda, 2019.
- **Idea**: A **monotone MLP** parameterises the cumulative hazard `Λ(τ)`; intensity is recovered by **autograd** on `τ`:

```
H_i = RNN([type_emb + τ])
Λ_m(τ) = MLP_m( [H_i, time_emb(τ)] )  (weights forced ≥ 0 → monotone in τ)
λ_m(τ) = ∂Λ_m / ∂τ                  (computed via torch.autograd.grad)
```

- **Code**: `torch_fullynn.py`.
  - `CumulHazardFunctionNetwork.init_weights_positive` clamps params to `≥ eps`.
  - `derivative_integral_lambda` is obtained per‑mark via `torch.autograd.grad`; this is why `TorchModelWrapper` keeps gradients enabled even during validation for FullyNN.
  - Log‑likelihood uses `Λ_m(τ).sum(-1)` directly as the non‑event integral (closed form, no MC needed).

- **Useful config**: `model_specs.num_mlp_layers`, `model_specs.proper_marked_intensities` (if True, computes per‑mark gradients; else shares one cumulative for all marks).

### 11.4 SAHP (ICML'20)

- **Paper**: *Self‑Attentive Hawkes Process*, Zhang et al., 2020.
- **Idea**: Transformer encoder + parametric decay between events:

```
H_i = TransformerEncoder( type_emb + time_shifted_pos_enc(t, τ) )      (Eq. 6)
λ*_m(t) = softplus( μ_m + (η_m − μ_m) · exp(−γ_m · (t − t_i)) )       (Eq. 15)
```

- **Code**: `torch_sahp.py`.
  - `TimeShiftedPositionalEncoding` (in `torch_baselayer.py`) implements Eq. 8.
  - `state_decay` realises Eq. 15 with `μ`, `η`, `γ` heads (each `Linear → GELU` or `Softplus`).
  - Predictions use thinning on top of `compute_intensities_at_sample_times`.

### 11.5 THP (ICML'20)

- **Paper**: *Transformer Hawkes Process*, Zuo et al., 2020.
- **Idea**: Pure transformer with an **affine** time decay:

```
H_i = TransformerEncoder( type_emb + TimePositionalEncoding(t) )       (Eq. 5)
λ*_m(t) = ScaledSoftplus( linear_m(H_i) + α_m · (t − t_i) + β_m )      (Eq. 6)
```

- **Code**: `torch_thp.py`. Pre‑norm style transformer, `factor_intensity_decay` and `factor_intensity_base` are per‑mark scalars (Xavier‑normal init). `compute_states_at_sample_times` broadcasts `α · τ_sample` over the grid for MC integration.

### 11.6 IntensityFree (ICLR'20)

- **Paper**: *Intensity‑Free Learning of TPPs*, Shchur et al., 2020.
- **Idea**: Skip the intensity. Model `p(τ | H)` as a **mixture of log‑normals**, and `p(k | H)` as a softmax. Likelihood is:

```
log L = Σ_i [ log p(τ_i | H_{i-1}) + log p(k_i | H_{i-1}) ]            (no integral!)
```

- **Code**: `torch_intensity_free.py`.
  - `LogNormalMixtureDistribution` wraps `torch.distributions.MixtureSameFamily` with an `AffineTransform(mean_log, std_log) ∘ ExpTransform` — those normalising stats come from `TPPDataset.get_dt_stats` (injected by `base_runner.py`).
  - `clamp_preserve_gradients` keeps log‑scales in `[−5, 3]` to avoid pathological gradients.
  - `predict_one_step_at_every_event` is **overridden**: it samples from the mixture (no thinning needed) and uses an independent softmax for marks.
- **Useful config**: `model_specs.num_mix_components` (3 by default, paper uses 64 for hard datasets).
- **Deep dive**: see `docs/IFTPP_Technical_Documentation.md` for a complete walkthrough including a space‑extension plan `p(t, x) = p(t)·p(x|t)`.

### 11.7 ODETPP (ICLR'21 simplified)

- **Paper**: *Neural Spatio‑Temporal Point Processes*, Chen, Amos & Nickel, 2021. The implementation in this repo is the **temporal‑only simplification**: between events the latent obeys a learned ODE; events apply an instantaneous jump.

```
dx/dt = MLP(x)   between events
x_{t_i+} = x_{t_i−} + Embed(k_i)      at events
λ*_m(t) = softplus( linear_m(x_t) )
```

- **Code**: `torch_ode_tpp.py`.
  - `NeuralODEAdjoint` implements adjoint‑sensitivity backprop manually (saves memory vs `torchdiffeq`).
  - `NeuralODE.forward` calls the fixed‑step `rk4_step_method` for `ode_num_sample_per_step` micro‑steps per inter‑event interval.
  - The model unrolls events with `torch.unbind(time_delta_seqs_, dim=-2)` — slow for long sequences but conceptually clear.

- **Useful config**: `model_specs.ode_num_sample_per_step`, `model_specs.time_factor` (scaling applied to `τ` to keep RK4 stable).

### 11.8 AttNHP (ICLR'22)

- **Paper**: *Transformer Embeddings of Irregularly Spaced Events and Their Participants*, Yang, Mei & Eisner, 2022.
- **Idea**: Stack of attention layers operating **at arbitrary query times**. Each head learns its own context; the model can evaluate intensity at any query time `s` by computing the attention between past events and a synthetic "query" token at `s`.

- **Code**: `torch_attnhp.py`.
  - `forward_pass` runs `n_head × n_layers` attention blocks; for each block the keys/values are the event sequence augmented with a "current layer" placeholder, and the query stream is shifted in time.
  - `compute_states_at_sample_times` reshapes the sample‑time axis into the batch dimension to evaluate intensity at many candidate times in parallel (very GPU‑friendly).

- **Useful config**: standard transformer knobs. The default `experiment_config.yaml::AttNHP_train` keeps `n_head=2`, `n_layers=2` for the small datasets.

### 11.9 ANHN (IJCNN'21, optional)

`torch_anhn.py` implements *Attentive Neural Hawkes Network*. It is registered in `easy_tpp.model.__init__` but is **not in the README model list** — keep it in mind if you want a relatively simple baseline that combines an LSTM with attention‑computed time decays.

### 11.10 WSMTHP (NeurIPS'24)

- **Paper**: *Is Score Matching Suitable for Estimating Point Processes?*, Zhao et al., 2024.
- **Idea**: Train a THP‑style encoder with **Weighted Score Matching** on the total intensity instead of log‑likelihood, which removes the need to estimate the integral term. Mark prediction uses a separate cross‑entropy on the per‑mark intensities.

```
score s(τ) = ∂ log λ_tot(τ) / ∂τ − λ_tot(τ)
L_WSM     = ∑ [ ½ h(τ_n) · s(τ_n)² + h(τ_n) · s'(τ_n) + h'(τ_n) · s(τ_n) ]
L_total   = L_WSM + CE_coef · L_CE  (+ optional finite‑window survival classifier)
```

- **Weight function** (`two_side_op`, the only currently supported choice):

```
h(τ_n)  = (T − t_{n−1}) / 2 − |t_n − (T + t_{n−1}) / 2|
h'(τ_n) = −1  if t_n > (T + t_{n−1})/2  else +1
```

- **Code**: `torch_wsm_thp.py` (~600 lines, the most elaborate model file).
  - `_EncoderLayer` — pre‑norm transformer block, reusing `MultiHeadAttention`.
  - `_temporal_enc` — model‑internal sinusoidal absolute‑time encoding (registered as a buffer).
  - `_get_intensity(τ, cond)` — Hawkes‑style intensity head `softplus(tanh(aff(cond)) · τ + base(cond))`.
  - `_compute_score` — total‑intensity score by `autograd.grad(log λ_tot, τ)` − `λ_tot`.
  - `loglike_loss` returns the **WSM loss** during training and falls back to `nll_loss` (standard MC log‑likelihood) at validation/test time, so EasyTPP can still log `loglike` and compute `rmse / acc`.
  - Optional `with_survival=True` adds a survival classifier (`logits → BCE` between "process continues" vs. "terminal"), used together with the finite‑window correction described in the paper.

- **Useful config** (under `model_specs`):

| Field               | Meaning                                                           |
|---------------------|-------------------------------------------------------------------|
| `h_type`            | `two_side_op` (only one currently supported)                       |
| `CE_coef`           | Weight on the mark cross‑entropy loss                              |
| `T_mode`            | How to set the observation window `T`: `manual` / `train_global` / `batch` |
| `max_observed_time` | Used iff `T_mode=manual`; auto‑filled for `train_global`           |
| `d_inner`           | FFN hidden dim (default `hidden_size * 2`)                         |
| `with_survival`, `alpha_survival`, `alpha_neg` | Optional survival classifier params |

> `RunnerConfig._maybe_set_max_observed_time` is the bridge that pre‑computes `T` from the training split when `T_mode='train_global'`. This is the only place where the data is touched *before* the runner is built.

### 11.11 S2P2 (NeurIPS'25)

- **Paper**: *Deep Continuous‑Time State‑Space Models for Marked Event Sequences*, anonymous OpenReview submission 74SvE2GZwW.
- **Idea**: A stack of *deep linear* continuous‑time SSM layers, each evolving its own diagonal latent `x_t ∈ ℂ^P`, communicating through residual streams `u_t ∈ ℝ^H` between layers. Intensity is obtained from the deepest residual stream by `softplus(linear(u_t))`.

```
For each layer l = 1..L:
    x_{t+} = E_l · α_t (impulse)  +  A_l x_{t−}  (between events)
    u_t   = LayerNorm( σ(C_l x_t + D_l u_t^{(l-1)}) + u_t^{(l-1)} )
λ*_m(t)   = softplus( W u_t^{(L)} )
```

- **Code**: `torch_s2p2.py`.
  - `ComplexEmbedding` — initialises both real and imaginary parts of the per‑mark complex embedding with a small scale.
  - `IntensityNet` — `Linear → ScaledSoftplus`.
  - `S2P2.forward` orchestrates all `L` `LLH` layers, returning `right_xs_BNLP` (right limits of latent `x` per layer) and `right_us_BNH` (right limits of residual stream per layer).
  - `loglike_loss` computes intensity at events using either the backward variant (`left_u_BNm1H` already available) or by manually evolving via `_get_intensity` + `_evolve_and_get_intensity_at_sampled_dts`.

- **Useful config** (mirrors the recommended values in `experiment_config.yaml::S2P2_train`):

| Field                | Meaning                                                          |
|----------------------|------------------------------------------------------------------|
| `P`                  | Latent state dimension (`x_t ∈ ℂ^P`)                              |
| `hidden_size` (H)    | Residual‑stream dimension (`u_t ∈ ℝ^H`)                           |
| `num_layers`         | Number of stacked LLH layers                                     |
| `act_func`           | `gelu` / `full_glu` / `half_glu`                                  |
| `for_loop`           | `True` to unroll (clearer, slower); `False` for parallel scan    |
| `pre_norm` / `post_norm` | Mostly fixed (False / True)                                  |
| `int_forward_variant` / `int_backward_variant` | Which ZOH variant of the integral form to use |
| `relative_time`      | Predict per‑interval rescaling factor (Sec. 3.3)                  |
| `dt_init_min/max`    | Range used to initialise `Δ`                                      |
| `dropout_rate`       | Dropout after activation, before LayerNorm                       |
| `complex_values`     | `True` for complex SSM (default), `False` for real‑only           |

> S2P2 is the only model that **replaces** `self.layer_type_emb` (set to `None` after `super().__init__`) with its own `layers_mark_emb`. Keep that in mind if you copy its skeleton.

---

## 12. Hyper‑parameter optimisation (`hpo`)

`easy_tpp/hpo/optuna_hpo.py` exposes a registrable `OptunaTuner` (`@HyperTuner.register('optuna')`). To use it:

1. Switch `pipeline_config_id` to `hpo_runner_config` (see `examples/configs/hpo_config.yaml`).
2. Add an `hpo:` block with `storage_uri`, `n_trials`, `framework_id: optuna`.
3. Use the magic syntax `suggest_int(low, high)`, `suggest_float(low, high)`, ... directly in any yaml value. The tuner intercepts those at trial creation.
4. Add a *trial function* in `easy_tpp/default_registers/register_optuna_trials.py` that maps `(trial, trainer_config, model_config) -> dict` of suggested overrides.

The objective is the metric returned by `runner.evaluate` (the runner returns `metric['rmse']` by default, so smaller is better — direction is inferred from `MetricsHelper.get_metric_direction`).

---

## 13. Metrics & evaluation conventions

`easy_tpp/default_registers/register_metrics.py` ships with two registered metrics:

| Name   | Direction | What it measures                                                              |
|--------|-----------|-------------------------------------------------------------------------------|
| `rmse` | minimise  | Root mean squared error of predicted inter‑event time vs ground truth `dtime` |
| `acc`  | maximise  | Accuracy of predicted next mark vs ground truth                                |

In addition, the runner internally tracks `loglike = -loss / num_events` and persists the **best** checkpoint by validation `loglike`. The `MetricsTracker` lives in `easy_tpp/utils/metrics.py`.

When you call `runner.run()` for `stage: gen`, the runner pickles the full prediction tensor (`pred_dtime`, `pred_type`, labels) to `pred.pkl` in the current working directory. That file is your starting point for any downstream evaluation (e.g. mark Top‑K, calibration, OTD distance).

---

## 14. End‑to‑end usage walkthroughs

### 14.1 Train an off‑the‑shelf model in three commands

```powershell
# 1. Install in editable mode (or `pip install easy-tpp`)
git clone https://github.com/ant-research/EasyTemporalPointProcess.git
cd EasyTemporalPointProcess
pip install -r requirements.txt
python setup.py develop

# 2. Pick one of the example experiments shipped in examples/configs/experiment_config.yaml
cd examples
python train_nhp.py --config_dir configs/experiment_config.yaml --experiment_id NHP_train

# 3. Evaluate / generate the trained model — change `stage: eval` or `stage: gen` and set
#    `pretrained_model_dir` to the saved checkpoint path inside checkpoints/.
python train_nhp.py --config_dir configs/experiment_config.yaml --experiment_id NHP_eval
```

### 14.2 Train a new dataset with an existing model

1. Drop your data in `data/<your_set>/` in either pickle or json/HF format (see §5.1).
2. Append a new top‑level entry to the `data:` block:

   ```yaml
   data:
     my_set:
       data_format: pkl
       train_dir: ../data/my_set/train.pkl
       valid_dir: ../data/my_set/dev.pkl
       test_dir: ../data/my_set/test.pkl
       data_specs:
         num_event_types: 7
         pad_token_id: 7   # always = num_event_types
         padding_side: right
   ```

3. Create an experiment with `dataset_id: my_set`.
4. Run `python train_nhp.py --experiment_id MyExp`.

### 14.3 Multi‑step generation

Set `base_config.stage: gen` and `model_config.thinning.num_step_gen: 20` (or whatever horizon). The runner will call `predict_multi_step_since_last_event` and dump `pred.pkl`. Reload it with `pickle.load(open('pred.pkl','rb'))` and inspect.

### 14.4 Hyperparameter sweep

Use `examples/train_nhp_hpo.py` as a template. Inside the trial function (registered in `default_registers/register_optuna_trials.py`) suggest hyper‑parameters, e.g.:

```python
@register_optuna_trial(exp_id='NHP_train')
def nhp_trial(trial, trainer_config=None, model_config=None):
    return {
        'model_config': {
            'hidden_size': trial.suggest_int('hidden_size', 16, 128, step=16),
        },
        'trainer_config': {
            'learning_rate': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        },
    }
```

Then `python train_nhp_hpo.py --config_dir configs/hpo_config.yaml --experiment_id NHP_train`.

---

## 15. Cheat‑sheet for adding a new model

```
1. Create easy_tpp/model/torch_model/torch_mymodel.py
   class MyModel(TorchBaseModel):
       def __init__(self, model_config): ...
       def loglike_loss(self, batch) -> (loss, num_events): ...
       def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs,
                                               type_seqs, sample_dtimes, **kwargs): ...

2. Register the class so `TorchBaseModel.generate_model_from_config` can find it:
       # easy_tpp/model/__init__.py
       from easy_tpp.model.torch_model.torch_mymodel import MyModel as TorchMyModel
       __all__.append('TorchMyModel')

3. Add an experiment to examples/configs/experiment_config.yaml using model_id: MyModel.
   Put your custom hyper-params under model_config.model_specs.

4. (Optional) override predict_one_step_at_every_event /
   predict_multi_step_since_last_event if your model does not have
   an intensity function (cf. IntensityFree).

5. (Optional) special pipeline hook: if you need data statistics *before*
   model construction (e.g. WSMTHP needs T from the training split), add
   the logic to RunnerConfig._maybe_set_max_observed_time-equivalent.

6. Write a quick test: tests/ contains examples (see test_*.py).
```

Key invariants you must preserve:

- Always read the 5‑tuple `(time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask)` and respect `pad_token_id`.
- Discard the prediction *at* `t_N` when computing log‑likelihood (use `[:, :-1]` for inputs, `[:, 1:]` for targets — every reference model does this).
- Implement `compute_intensities_at_sample_times` to handle both the full sequence and `compute_last_step_only=True` (needed by thinning's multi‑step generation).

---

## 16. Cheat‑sheet for adding a new dataset

1. Convert your raw event log into either the Gatech pkl format or the JSON columns described in §5.1. Use `examples/script_data_processing/` as a starting point if your data is in raw csv/parquet.
2. Make sure timestamps are **strictly increasing** and the first `time_delta` is `0`.
3. The first and last marks can be padded marks (`pad_token_id`) if they represent the observation window endpoints — see how `RMTPP.forward` documents the convention.
4. Add the dataset block to your yaml as in §14.2; set `num_event_types` to the **actual** number of marks and `pad_token_id = num_event_types`.
5. Optional: if your sequences are very long, set `data_specs.padding_strategy: max_length` and `data_specs.max_len: <some int>` (and a matching `truncation_strategy: longest_first`) to bound memory.

---

## 17. Known pitfalls / common bugs

- **The first inter‑event time is zero by convention.** Slicing usually drops it (`time_delta_seqs[:, 1:]`). If you forget, you will silently compute log‑likelihood on a degenerate interval.
- **`num_event_types` excludes padding** but `num_event_types_pad` includes it. Most layers use one or the other consistently — copy from a reference model rather than re‑deriving.
- **Attention mask convention is inverted** vs HuggingFace (`1 = mask`, `0 = keep`). See `MultiHeadAttention.attention` in `torch_baselayer.py`.
- **FullyNN needs gradients at validation** (`torch_wrapper.py:116`). If you fork it, do not unconditionally wrap validation in `torch.no_grad()`.
- **WSMTHP requires `T`** before the model is built. The yaml field is `model_specs.T_mode`. If you wire it manually, remember `RunnerConfig._maybe_set_max_observed_time` is the only place this is computed.
- **MPS backend can silently produce wrong gradients** for S2P2 and other models with complex tensors. The provided config recommends CPU or CUDA (`gpu: -1` or `gpu: 0`).
- **`ScaledSoftplus` clamps at `log(1e5)`** — if your intensities saturate, normalise your timestamps first instead of fighting the clamp.
- **`base_runner.py:43` blindly calls `train_loader.dataset.get_dt_stats()`** — make sure every dataset you add has more than one event per sequence (the helper assumes `dts[1:]` is non‑empty).
- **HuggingFace dataset loading requires internet access** the first time. Pre‑download with `datasets.load_dataset('easytpp/taxi')` if you train offline.

---

## 18. Glossary of tensor shape symbols

Across the codebase the following short‑hand suffixes appear in variable names (especially in `S2P2` and `WSMTHP`):

| Letter | Meaning                                          |
|--------|--------------------------------------------------|
| `B`    | Batch size                                       |
| `N`    | Sequence length (number of events incl. padding) |
| `M`    | Number of marks (`num_event_types`)              |
| `H`    | Hidden size / residual stream dim                |
| `P`    | Latent state dim (S2P2 only)                     |
| `L`    | Number of stacked SSM layers (S2P2 only)         |
| `G`    | Number of grid points for the loss integral     |
| `S`    | Number of thinning candidate samples             |

Common abbreviations: `dts` = inter‑event times, `marks` = event types, `cond` = history encoding, `ll` = log‑likelihood, `lambda_at_event` = intensity evaluated *at the actual event*, `lambdas_loss_samples` = intensities evaluated at grid points used to approximate the integral.

---

### Where to go next

- **Read `torch_basemodel.py` cover to cover** — the rest of the codebase is much easier afterwards.
- **Pair `torch_nhp.py` with the original NHP paper** — it is the cleanest mapping from equations to code and a great reading template for the other models.
- **For your own research**: clone `torch_thp.py` or `torch_intensity_free.py` (depending on whether your method works in intensity space or distribution space) and follow the §15 checklist.
- For an even deeper IFTPP walkthrough (loss derivation, thinning vs sampling, spatial extension `p(t,x)=p(t)·p(x|t)`), see the companion document `docs/IFTPP_Technical_Documentation.md`.

Happy hacking, and may your inter‑event times be log‑normal! 🎲
