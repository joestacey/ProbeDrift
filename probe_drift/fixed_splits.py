"""
Fixed benchmark splits bundled with the library.

Split files are gzip-compressed JSON stored in probe_drift/benchmark_splits/
and committed to the repository.

Naming convention:
    eval__{eval_dataset}.json.gz
    train__{eval_dataset}__{ood_setting}.json.gz

Format (version 2):
    {
      "version": 2,
      "dataset_key": "sciq",       // eval splits only; training splits omit this
      "few_shot_meta": { ... },    // raw few-shot rows; only for trivia_qa, med_quad, mmlu
      "examples": [ { ... }, ... ] // raw field dicts, one per example
    }

    Raw fields per dataset type (no pre-formatted strings anywhere):
      xsum / cnn_dailymail / samsum:  {"text": "...", "y": "..."}
      sciq:       {"question": "...", "context": "...", "y": "..."}
      trivia_qa:  {"question": "...", "y": ["alias1", ...]}
      pubmed_qa:  {"contexts": [...], "question": "...", "y": "..."}
      qa (CoQA):  {"x": "...", "y": "..."}  (pass-through; pre-formatted turn, no story grouping)
      truthful_qa: {"question": "...", "y": ["answer1", ...]}
      med_quad:   {"question": "...", "y": "..."}
      mmlu:       {"question": "...", "choices": [...], "subject": "...", "y": "A/B/C/D"}

    Training splits have one additional field per example:
      {"dataset_key": "samsum", "text": "...", "y": "..."}

    few_shot_meta raw fields (formatted at load time, never stored pre-formatted):
      trivia_qa:  [{"question": "...", "y": "answer_string"}, ...]
      med_quad:   [{"question": "...", "y": "answer_string"}, ...]
      mmlu:       {"subject1": [{"question": "...", "choices": [...], "y": "A", ...}], ...}

Changing a prompt in dataset_configs.py takes effect immediately at load time for
all existing split files: no regeneration required.
"""

import os
import json
import gzip

from probe_drift.dataset_configs import DATASET_CONFIG

_SPLITS_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_splits')


def _train_path(eval_dataset, ood_setting):
    return os.path.join(_SPLITS_DIR, f'train__{eval_dataset}__{ood_setting}.json.gz')


def _eval_path(eval_dataset):
    return os.path.join(_SPLITS_DIR, f'eval__{eval_dataset}.json.gz')


def has_train_split(eval_dataset, ood_setting):
    return os.path.exists(_train_path(eval_dataset, ood_setting))


def has_eval_split(eval_dataset):
    return os.path.exists(_eval_path(eval_dataset))


def splits_support_instruct(eval_dataset):
    """Return True only if the stored eval split is version 2 (raw fields)."""
    path = _eval_path(eval_dataset)
    if not os.path.exists(path):
        return False
    data = _load_gz(path)
    return data.get('version', 1) >= 2


def _load_gz(path):
    with gzip.open(path, 'rb') as f:
        return json.loads(f.read().decode('utf-8'))


def _hf_name(cfg):
    """Return the HuggingFace dataset name for pattern matching in _format_raw_rows."""
    p = cfg['dataset_path']
    return p[0] if isinstance(p, list) else p


def _format_split(data, dataset_key, instruct):
    """
    Apply prompt formatting to a loaded version:2 split dict.
    Returns (x, y) lists.
    """
    from probe_drift.dataset import Dataset

    cfg_key = dataset_key + ('_instruct' if instruct else '')
    cfg = DATASET_CONFIG[cfg_key]

    examples     = data['examples']
    few_shot_meta = data.get('few_shot_meta')

    x, y = Dataset._format_raw_rows(
        examples, _hf_name(cfg), cfg, instruct=instruct, few_shot_meta=few_shot_meta,
    )
    return x, y


def load_train_split(eval_dataset, ood_setting, instruct=False):
    """Return {'x': [...], 'y': [...]} for the fixed benchmark training split."""
    data = _load_gz(_train_path(eval_dataset, ood_setting))

    if data.get('version', 1) == 1:
        return {'x': data['x'], 'y': data['y']}

    # version 2: examples carry per-row dataset_key; format each source separately
    # then reassemble in original order to preserve the stored shuffle.
    from probe_drift.dataset import Dataset

    examples   = data['examples']
    x_out      = [''] * len(examples)
    y_out      = [None] * len(examples)
    source_ids = [row['dataset_key'] for row in examples]

    # Group by dataset_key, keeping track of original positions.
    from collections import defaultdict
    groups = defaultdict(list)  # dataset_key -> [(original_index, row)]
    for i, row in enumerate(examples):
        groups[row['dataset_key']].append((i, row))

    few_shot_meta_all = data.get('few_shot_meta', {})

    for dk, indexed_rows in groups.items():
        positions = [i for i, _ in indexed_rows]
        rows      = [r for _, r in indexed_rows]
        cfg_key   = dk + ('_instruct' if instruct else '')
        cfg       = DATASET_CONFIG[cfg_key]
        fsm       = few_shot_meta_all.get(dk)
        x_part, y_part = Dataset._format_raw_rows(
            rows, _hf_name(cfg), cfg, instruct=instruct, few_shot_meta=fsm,
        )
        for pos, xi, yi in zip(positions, x_part, y_part):
            x_out[pos] = xi
            y_out[pos] = yi

    return {'x': x_out, 'y': y_out, 'source_ids': source_ids}


def load_eval_split(eval_dataset, instruct=False):
    """Return {'x': [...], 'y': [...]} for the fixed benchmark eval split."""
    data = _load_gz(_eval_path(eval_dataset))

    if data.get('version', 1) == 1:
        return data

    x, y = _format_split(data, eval_dataset, instruct)
    return {'x': x, 'y': y}


