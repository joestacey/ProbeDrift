"""
Structural tests for OOD training specifications.

Verifies that get_training_spec() returns correct dataset compositions for all
(eval_dataset, ood_setting) combinations.  No data loading or external
dependencies required; these tests ship with the library.

Run:
    python -m pytest tests/test_ood_settings.py -v
"""

import pytest

from probe_drift.dataset_configs import VALID_EVAL_DATASETS
from probe_drift.ood_settings import get_training_spec, OOD_SETTINGS_FULL


@pytest.mark.parametrize("eval_ds,ood_setting", [
    (ds, ood) for ds in VALID_EVAL_DATASETS for ood in OOD_SETTINGS_FULL
])
def test_training_spec_total_samples(eval_ds, ood_setting):
    """Every (eval_dataset, ood_setting) combination sums to exactly 1800 training samples."""
    spec = get_training_spec(eval_ds, ood_setting, instruct=False)
    total = sum(n for _, n in spec)
    assert total == 1800, f"{eval_ds} x {ood_setting}: total={total}"


@pytest.mark.parametrize("eval_ds,ood_setting", [
    (ds, ood) for ds in VALID_EVAL_DATASETS for ood in OOD_SETTINGS_FULL
])
def test_training_spec_no_eval_overlap_for_ood(eval_ds, ood_setting):
    """OOD settings never train on the eval dataset itself."""
    if ood_setting == 'ID':
        pytest.skip("ID intentionally trains on the eval dataset")
    spec = get_training_spec(eval_ds, ood_setting, instruct=False)
    train_datasets = [k for k, _ in spec]
    assert eval_ds not in train_datasets, \
        f"{eval_ds} appears as a training dataset in {ood_setting}"


@pytest.mark.parametrize("eval_ds", VALID_EVAL_DATASETS)
def test_loo_uses_exactly_9_datasets(eval_ds):
    """OOD_LEAVE_ONE_OUT always trains on exactly 9 datasets, 200 samples each."""
    spec = get_training_spec(eval_ds, 'OOD_LEAVE_ONE_OUT')
    assert len(spec) == 9, f"{eval_ds} LOO has {len(spec)} datasets, expected 9"
    assert all(n == 200 for _, n in spec), "Not all LOO datasets have 200 samples"


@pytest.mark.parametrize("eval_ds", ['sciq', 'trivia_qa', 'qa', 'pubmed_qa'])
def test_same_task_qa_trains_on_med_quad(eval_ds):
    """OOD_ONE_DATASET_SAME_TASK for QA datasets trains exclusively on med_quad."""
    spec = get_training_spec(eval_ds, 'OOD_ONE_DATASET_SAME_TASK')
    assert spec == [('med_quad', 1800)], f"Expected [('med_quad', 1800)], got {spec}"


@pytest.mark.parametrize("eval_ds", ['xsum', 'cnn_dailymail'])
def test_same_task_sum_trains_on_samsum(eval_ds):
    """OOD_ONE_DATASET_SAME_TASK for summarisation datasets trains exclusively on samsum."""
    spec = get_training_spec(eval_ds, 'OOD_ONE_DATASET_SAME_TASK')
    assert spec == [('samsum', 1800)], f"Expected [('samsum', 1800)], got {spec}"


@pytest.mark.parametrize("eval_ds", ['sciq', 'trivia_qa', 'qa', 'pubmed_qa'])
def test_diff_task_qa_trains_on_all_summarisation(eval_ds):
    """OOD_DIFF_TASK for QA datasets trains on all three summarisation datasets."""
    spec = get_training_spec(eval_ds, 'OOD_DIFF_TASK')
    keys = [k for k, _ in spec]
    assert sorted(keys) == sorted(['samsum', 'xsum', 'cnn_dailymail']), \
        f"{eval_ds} OOD_DIFF_TASK: {keys}"
    assert all(n == 600 for _, n in spec)


@pytest.mark.parametrize("eval_ds", ['xsum', 'cnn_dailymail'])
def test_diff_task_sum_trains_on_6_qa_datasets(eval_ds):
    """OOD_DIFF_TASK for summarisation datasets trains on 6 QA datasets (pubmed_qa excluded)."""
    spec = get_training_spec(eval_ds, 'OOD_DIFF_TASK')
    keys = [k for k, _ in spec]
    assert 'pubmed_qa' not in keys, "pubmed_qa should be excluded from OOD_DIFF_TASK for summarisation"
    assert len(keys) == 6, f"Expected 6 QA datasets, got {len(keys)}: {keys}"
    assert all(n == 300 for _, n in spec)
