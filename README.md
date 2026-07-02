# ProbeDrift

**ProbeDrift** is a Python library for evaluating the out-of-distribution (OOD) generalisation of supervised uncertainty quantification (UQ) probes for large language models. It provides a simple API for loading the training and evaluation datasets required to benchmark how well a UQ probe trained on one dataset or task type transfers to others.

---

## Benchmarks

### ProbeDrift (full benchmark)

Evaluates on six datasets spanning two output-length regimes:

| Form | Dataset | Task |
|------|---------|------|
| Short-form | SciQ | QA |
| Short-form | TriviaQA | QA |
| Short-form | CoQA (`'qa'`) | QA |
| Long-form | PubMedQA | QA |
| Long-form | XSum | Summarisation |
| Long-form | CNN/DailyMail | Summarisation |

Five OOD conditions are evaluated:

| Setting | Description | # Training datasets | Samples per dataset | Total |
|---------|-------------|-------------------|-------------------|-------|
| `ID` | Train and evaluate on the same dataset | 1 | 1800 | 1800 |
| `OOD_LEAVE_ONE_OUT` | Train on all 9 other datasets | 9 | 200 | 1800 |
| `OOD_ONE_DATASET_SAME_TASK` | Train on one dataset from the same task type | 1 | 1800 | 1800 |
| `OOD_DIFF_TASK` | Train on all datasets from the opposite task type | 3 or 6 | 600 or 300 | 1800 |
| `OOD_ONE_DATASET_DIFF_TASK` | Train on one dataset from the opposite task type | 1 | 1800 | 1800 |

### ProbeDriftLight (lightweight benchmark)

A reduced benchmark for rapid evaluation. Uses three datasets and three OOD conditions:

| Form | Dataset | Task |
|------|---------|------|
| Short-form | SciQ | QA |
| Short-form | TriviaQA | QA |
| Long-form | PubMedQA | QA |

OOD conditions: `ID`, `OOD_LEAVE_ONE_OUT`, `OOD_DIFF_TASK`

Both `ProbeDrift` and `ProbeDriftLight` can be run with base or instruction-tuned models — simply pass `instruct=True` to use instruct-formatted prompts.

---

## Installation

```bash
pip install git+https://github.com/joestacey/ProbeDrift.git
```

Dependencies: `numpy`

---

## Quick Start

### High-level API: `get_eval_suite`

Returns all datasets for a benchmark and OOD setting at once, grouped into short-form and long-form sub-dictionaries.

```python
from probe_drift import get_eval_suite, OOD_SETTINGS_FULL

for ood_setting in OOD_SETTINGS_FULL:
    suite = get_eval_suite(
        eval_mode='ProbeDrift',   # full benchmark
        ood_setting=ood_setting,
        instruct=False,
        batch_size=1,
    )
    for form in ['short', 'long']:
        for dataset_name, (train_dataset, eval_dataset) in suite[form].items():
            print(f"{ood_setting} | {form} | {dataset_name}: "
                  f"train={len(train_dataset.x)}, eval={len(eval_dataset.x)}")
            # pass train_dataset and eval_dataset to your probe training / eval pipeline
```

For the lightweight benchmark:

```python
from probe_drift import get_eval_suite, OOD_SETTINGS_REDUCED

for ood_setting in OOD_SETTINGS_REDUCED:
    suite = get_eval_suite(
        eval_mode='ProbeDriftLight',
        ood_setting=ood_setting,
    )
```

For instruction-tuned models, add `instruct=True` to either benchmark:

```python
suite = get_eval_suite(
    eval_mode='ProbeDrift',         # or 'ProbeDriftLight'
    ood_setting='OOD_LEAVE_ONE_OUT',
    instruct=True,
)
```

### Low-level API: `get_datasets`

Returns a single `(train_dataset, eval_dataset)` pair for one dataset and OOD condition.

```python
from probe_drift import get_datasets

train_dataset, eval_dataset = get_datasets(
    eval_dataset='sciq',
    ood_setting='OOD_LEAVE_ONE_OUT',
    instruct=False,
    batch_size=1,
)

print(f"Training examples: {len(train_dataset.x)}")  # 1800
print(f"Eval examples:     {len(eval_dataset.x)}")   # up to 2000
```

### Single-dataset mode

Pass an individual dataset name as `eval_mode` to evaluate one dataset across all five OOD conditions:

```python
suite = get_eval_suite(
    eval_mode='pubmed_qa',
    ood_setting='OOD_ONE_DATASET_SAME_TASK',
)
```

---

## Dataset objects

Both functions return `Dataset` objects with the following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.x` | `List[str]` | Formatted input prompts |
| `.y` | `List[str]` | Target answers |
| `.source_ids` | `List[str]` | Source dataset name per example. For training datasets this varies across examples; for eval datasets all entries are the same eval dataset name. Use this to configure per-dataset generation settings (e.g. `max_new_tokens`) in your pipeline. |
| `.batch_size` | `int` | Batch size |

And methods: `__iter__`, `__len__`, `subsample`, `concat`, `select`.

---

## API Reference

### `get_eval_suite(eval_mode, ood_setting, ...)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_mode` | str | — | `'ProbeDrift'`, `'ProbeDriftLight'`, or one of the 6 dataset names |
| `ood_setting` | str | — | One of `OOD_SETTINGS_FULL` or `OOD_SETTINGS_REDUCED` |
| `instruct` | bool | `False` | Use instruct prompt variants (for instruction-tuned models) |
| `batch_size` | int | `1` | Batch size for returned Dataset objects |
| `subsample_eval` | int | `2000` | Max eval examples to keep. Set `-1` for the full eval split. |

**Returns:** `{'short': {dataset_name: (train_dataset, eval_dataset), ...}, 'long': {...}}`

### `get_datasets(eval_dataset, ood_setting, ...)`

Same optional parameters as above, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_dataset` | str | — | One of: `'sciq'`, `'trivia_qa'`, `'qa'`, `'pubmed_qa'`, `'xsum'`, `'cnn_dailymail'` |

**Returns:** `(train_dataset, eval_dataset)`

---

## Available constants

```python
from probe_drift import (
    OOD_SETTINGS_FULL,           # all 5 OOD setting names (ProbeDrift)
    OOD_SETTINGS_REDUCED,        # 3 OOD setting names (ProbeDriftLight)
    PROBE_DRIFT_DATASETS,        # 6 dataset names in the full benchmark
    PROBE_DRIFT_LIGHT_DATASETS,  # 3 dataset names in the light benchmark
    VALID_EVAL_DATASETS,         # all 6 supported eval dataset names
    SHORT_FORM_DATASETS,         # ['sciq', 'trivia_qa', 'qa']
    LONG_FORM_DATASETS,          # ['pubmed_qa', 'xsum', 'cnn_dailymail']
)
```

---

## Reproducibility

ProbeDrift is designed so that results are fully reproducible:

- **Eval set** is always the same examples across all experiments (subsampled to at most `subsample_eval`).
- **Training set** order and composition are fixed: the bundled split files are pre-shuffled at generation time, so every load returns examples in the same order.
- Prompts are applied at load time from `dataset_configs.py` and include dataset-specific formatting: few-shot examples, multi-turn CoQA conversations, per-subject MMLU few-shot, etc.
- Training data size is fixed at **1800 examples** for every OOD condition.

---

## Customising the benchmark

The split files store raw source fields rather than pre-formatted strings. Prompts are applied at load time from `probe_drift/dataset_configs.py`, so changing a prompt there takes effect immediately without regenerating the splits.

The `source_ids` attribute on both training and eval datasets tells your pipeline which dataset each example comes from, so you can apply whatever per-dataset settings are appropriate for your model and task (e.g. generation length).

---

## Supported datasets

### Evaluation datasets

| Key | Source (HF path) | Form | Task |
|-----|-----------------|------|------|
| `sciq` | `sciq` | Short | QA |
| `trivia_qa` | `trivia_qa / rc.nocontext` | Short | QA |
| `qa` | `coqa` | Short | QA |
| `pubmed_qa` | `bigbio/pubmed_qa` | Long | QA |
| `xsum` | `xsum` | Long | Summarisation |
| `cnn_dailymail` | `cnn_dailymail / 3.0.0` | Long | Summarisation |

### Training-only datasets (OOD sources)

`med_quad`, `samsum`, `mmlu`, `truthful_qa` — plus instruct variants of all datasets above.
