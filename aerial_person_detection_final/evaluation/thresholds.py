"""Validation threshold selection for the locked test protocol."""
from evaluation.metrics import summarize


def filter_records(records, confidence_threshold):
    """Copy records while retaining predictions at or above the threshold."""
    return [
        {
            **record,
            'predictions': [p for p in record['predictions'] if p.confidence >= confidence_threshold],
        }
        for record in records
    ]


def select_f1_threshold(records, thresholds=None):
    """Select the threshold with maximum validation F1.

    Ties are resolved by choosing the higher threshold, reducing false alarms
    when two operating points have the same F1.
    """
    thresholds = thresholds or [i / 100 for i in range(5, 100, 5)]
    candidates = []
    for threshold in thresholds:
        score = summarize(filter_records(records, threshold), class_ids=[0])
        candidates.append((score['f1'], threshold, score))
    best_f1, best_threshold, best_score = max(candidates, key=lambda item: (item[0], item[1]))
    return {
        'validation_selected_threshold': best_threshold,
        'threshold_selection_metric': 'F1',
        'threshold_locked_before_test': True,
        'validation_f1': best_f1,
        'validation_operating_point': best_score,
        'candidate_thresholds': list(thresholds),
    }
