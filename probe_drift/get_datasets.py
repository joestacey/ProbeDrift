"""
Main API for loading training and evaluation datasets.
"""

from typing import Tuple, Dict

from probe_drift.dataset import Dataset
from probe_drift.dataset_configs import (
    VALID_EVAL_DATASETS,
    SHORT_FORM_DATASETS,
)
from probe_drift.ood_settings import (
    get_training_spec,
    REDUCED_DATASETS,
    FULL_DATASETS,
    OOD_SETTINGS_REDUCED,
    OOD_SETTINGS_FULL,
)
from probe_drift.fixed_splits import (
    has_train_split,
    splits_support_instruct,
    load_train_split,
    load_eval_split,
)


def get_datasets(
    eval_dataset: str,
    ood_setting: str,
    instruct: bool = False,
    batch_size: int = 1,
    subsample_eval: int = 2000,
) -> Tuple[Dataset, Dataset]:
    """
    Load one (train_dataset, eval_dataset) pair from the bundled benchmark splits.

    Parameters
    ----------
    eval_dataset : str
        One of: 'sciq', 'trivia_qa', 'qa', 'pubmed_qa', 'xsum', 'cnn_dailymail'
    ood_setting : str
        One of: 'ID', 'OOD_LEAVE_ONE_OUT', 'OOD_ONE_DATASET_SAME_TASK',
                'OOD_DIFF_TASK', 'OOD_ONE_DATASET_DIFF_TASK'
    instruct : bool
        If True, uses instruct prompt variants for all datasets.
    batch_size : int
        Batch size for the returned Dataset objects.
    subsample_eval : int
        Max eval examples to keep (fixed seed=1 for reproducibility).
        Set -1 to use the full eval split.

    Returns
    -------
    (train_dataset, eval_dataset) : Tuple[Dataset, Dataset]
    """
    base_eval = eval_dataset.replace('_instruct', '')

    if base_eval not in VALID_EVAL_DATASETS:
        raise ValueError(
            f"Unknown eval_dataset '{eval_dataset}'. "
            f"Must be one of: {VALID_EVAL_DATASETS}"
        )
    if not has_train_split(base_eval, ood_setting):
        raise ValueError(
            f"No bundled split for eval_dataset='{base_eval}', ood_setting='{ood_setting}'."
        )
    if instruct and not splits_support_instruct(base_eval):
        raise ValueError(
            f"Bundled split for '{base_eval}' does not support instruct mode."
        )

    train_data = load_train_split(base_eval, ood_setting, instruct=instruct)
    train_source_ids = train_data.get('source_ids')
    if train_source_ids is None:
        # v1 files: reconstruct source_ids from training spec (source-grouped order)
        training_spec = get_training_spec(base_eval, ood_setting, instruct=instruct)
        train_source_ids = [key for key, n in training_spec for _ in range(n)]
    train_ds = Dataset(
        train_data['x'], train_data['y'], batch_size,
        source_ids=train_source_ids,
    )

    eval_data = load_eval_split(base_eval, instruct=instruct)
    eval_ds = Dataset(
        eval_data['x'], eval_data['y'], batch_size,
        source_ids=[base_eval] * len(eval_data['x']),
    )
    if subsample_eval > 0:
        eval_ds.subsample(subsample_eval, seed=1)
    return train_ds, eval_ds


def get_eval_suite(
    eval_mode: str,
    ood_setting: str,
    instruct: bool = False,
    batch_size: int = 1,
    subsample_eval: int = 2000,
) -> Dict[str, Dict[str, Tuple[Dataset, Dataset]]]:
    """
    Load all datasets for a given eval mode and OOD setting, grouped by
    output-length form (short / long).

    Parameters
    ----------
    eval_mode : str
        'ProbeDriftLight': SciQ, TriviaQA, PubmedQA  (3 OOD settings)
        'ProbeDrift':      SciQ, TriviaQA, CoQA, PubmedQA, XSum, CNN/DailyMail
                            (5 OOD settings, full benchmark)
        Or one of the 6 individual dataset names for a single-dataset run
        with all 5 OOD settings available.
    ood_setting : str
        One of OOD_SETTINGS_FULL. OOD_ONE_DATASET_SAME_TASK and
        OOD_ONE_DATASET_DIFF_TASK are not valid for ProbeDriftLight.
    instruct : bool
        If True, uses instruct prompt variants. Intended for use with
        eval_mode='ProbeDriftLight' for instruction-tuned models.
    batch_size, subsample_eval :
        Passed through to get_datasets().

    Returns
    -------
    dict with keys 'short' and 'long', each mapping dataset_name to
    (train_dataset, eval_dataset).

    Example
    -------
    suite = get_eval_suite('ProbeDrift', 'OOD_LEAVE_ONE_OUT')
    for form in ['short', 'long']:
        for name, (train_ds, eval_ds) in suite[form].items():
            run_experiment(train_ds, eval_ds)
    """
    if eval_mode == 'ProbeDriftLight':
        datasets       = REDUCED_DATASETS
        valid_settings = OOD_SETTINGS_REDUCED
    elif eval_mode == 'ProbeDrift':
        datasets       = FULL_DATASETS
        valid_settings = OOD_SETTINGS_FULL
    elif eval_mode in VALID_EVAL_DATASETS:
        datasets       = [eval_mode]
        valid_settings = OOD_SETTINGS_FULL
    else:
        raise ValueError(
            f"Unknown eval_mode '{eval_mode}'. "
            f"Must be 'ProbeDrift', 'ProbeDriftLight', or one of {VALID_EVAL_DATASETS}"
        )

    if ood_setting not in valid_settings:
        raise ValueError(
            f"ood_setting '{ood_setting}' is not valid for eval_mode='{eval_mode}'. "
            f"Valid settings: {valid_settings}"
        )

    result = {'short': {}, 'long': {}}

    for ds_name in datasets:
        train_ds, eval_ds = get_datasets(
            eval_dataset=ds_name,
            ood_setting=ood_setting,
            instruct=instruct,
            batch_size=batch_size,
            subsample_eval=subsample_eval,
        )
        if ds_name in SHORT_FORM_DATASETS:
            result['short'][ds_name] = (train_ds, eval_ds)
        else:
            result['long'][ds_name] = (train_ds, eval_ds)

    return result
