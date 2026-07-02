"""
OOD setting definitions and training dataset resolution.

Maps the five evaluation settings to their concrete training dataset lists.
"""

from probe_drift.dataset_configs import (
    VALID_EVAL_DATASETS,
    QA_DATASETS,
    ALL_TRAIN_DATASETS,
)

# ---------------------------------------------------------------------------
# Exported constants
# ---------------------------------------------------------------------------

PROBE_DRIFT_LIGHT_DATASETS = ['sciq', 'trivia_qa', 'pubmed_qa']
PROBE_DRIFT_DATASETS       = ['sciq', 'trivia_qa', 'qa', 'pubmed_qa', 'xsum', 'cnn_dailymail']

# Aliases kept for internal use
REDUCED_DATASETS = PROBE_DRIFT_LIGHT_DATASETS
FULL_DATASETS    = PROBE_DRIFT_DATASETS

OOD_SETTINGS_REDUCED = [
    'ID',
    'OOD_LEAVE_ONE_OUT',
    'OOD_DIFF_TASK',
]

OOD_SETTINGS_FULL = [
    'ID',
    'OOD_LEAVE_ONE_OUT',
    'OOD_ONE_DATASET_SAME_TASK',
    'OOD_DIFF_TASK',
    'OOD_ONE_DATASET_DIFF_TASK',
]

# ---------------------------------------------------------------------------
# Training spec resolution
# ---------------------------------------------------------------------------

def get_training_spec(eval_dataset, ood_setting, instruct=False):
    """
    Return a list of (dataset_key, n_samples) tuples describing which datasets
    to load for training and how many samples to draw from each.

    Total samples always sums to 1800.

    Parameters
    ----------
    eval_dataset : str
        Base dataset name (without _instruct suffix).
    ood_setting : str
        One of OOD_SETTINGS_FULL.
    instruct : bool
        If True, appends '_instruct' to all returned dataset keys.

    Returns
    -------
    list of (str, int)
    """
    if eval_dataset not in VALID_EVAL_DATASETS:
        raise ValueError(
            f"Unknown eval_dataset '{eval_dataset}'. "
            f"Must be one of: {VALID_EVAL_DATASETS}"
        )
    if ood_setting not in OOD_SETTINGS_FULL:
        raise ValueError(
            f"Unknown ood_setting '{ood_setting}'. "
            f"Must be one of: {OOD_SETTINGS_FULL}"
        )

    suffix = '_instruct' if instruct else ''

    def key(base):
        return base + suffix

    is_qa = eval_dataset in QA_DATASETS

    if ood_setting == 'ID':
        return [(key(eval_dataset), 1800)]

    elif ood_setting == 'OOD_LEAVE_ONE_OUT':
        # All 10 datasets except the eval one, 200 samples each = 1800 total.
        return [(key(d), 200) for d in ALL_TRAIN_DATASETS if d != eval_dataset]

    elif ood_setting == 'OOD_ONE_DATASET_SAME_TASK':
        # One dataset from the same task family: QA -> med_quad, Sum -> samsum
        if is_qa:
            return [(key('med_quad'), 1800)]
        else:
            return [(key('samsum'), 1800)]

    elif ood_setting == 'OOD_DIFF_TASK':
        # All datasets from the opposite task family
        # QA  -> 3 summarisation datasets x 600 = 1800
        # Sum -> 6 QA datasets x 300 = 1800 (pubmed_qa excluded)
        if is_qa:
            return [
                (key('samsum'),        600),
                (key('xsum'),          600),
                (key('cnn_dailymail'), 600),
            ]
        else:
            return [
                (key('qa'),          300),
                (key('trivia_qa'),   300),
                (key('mmlu'),        300),
                (key('truthful_qa'), 300),
                (key('med_quad'),    300),
                (key('sciq'),        300),
            ]

    elif ood_setting == 'OOD_ONE_DATASET_DIFF_TASK':
        # One dataset from the opposite task family: QA -> samsum, Sum -> med_quad
        if is_qa:
            return [(key('samsum'),   1800)]
        else:
            return [(key('med_quad'), 1800)]
