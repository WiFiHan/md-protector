"""Detection metrics. Labels: 0 = machine, 1 = human; a higher score means more human-like."""
import numpy as np
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve


def best_threshold_by_avg_rec(y_true, y_score):
    """Threshold that maximizes AvgRec; predictions are human when score >= threshold."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_score.size == 0:
        return 0.5, 0.0

    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_labels = y_true[order]
    total_human = int((sorted_labels == 1).sum())
    total_machine = int((sorted_labels == 0).sum())

    # A threshold above the maximum score predicts everything as machine.
    best_threshold = float(sorted_scores[0]) + 1e-12
    best_avg_rec = 0.5 * (1.0 if total_machine > 0 else 0.0)

    human_tp, machine_fp, idx, n = 0, 0, 0, len(sorted_scores)
    while idx < n:
        score = sorted_scores[idx]
        next_idx = idx
        while next_idx < n and sorted_scores[next_idx] == score:
            if sorted_labels[next_idx] == 1:
                human_tp += 1
            else:
                machine_fp += 1
            next_idx += 1
        human_rec = human_tp / total_human if total_human > 0 else 0.0
        machine_rec = (total_machine - machine_fp) / total_machine if total_machine > 0 else 0.0
        avg_rec = 0.5 * (human_rec + machine_rec)
        if avg_rec > best_avg_rec + 1e-12:
            best_avg_rec = avg_rec
            best_threshold = float(score)
        idx = next_idx
    return best_threshold, best_avg_rec


def compute_metrics(labels, scores, threshold=None):
    """All metrics in percent. If threshold is None, the AvgRec-maximizing threshold is used."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if threshold is None:
        threshold, _ = best_threshold_by_avg_rec(labels, scores)
    preds = (scores >= threshold).astype(np.int64)

    n_human, n_machine = int(np.sum(labels == 1)), int(np.sum(labels == 0))
    human_rec = int(np.sum((labels == 1) & (preds == 1))) * 100 / n_human if n_human else 0.0
    machine_rec = int(np.sum((labels == 0) & (preds == 0))) * 100 / n_machine if n_machine else 0.0

    fpr, tpr, _ = roc_curve(labels, scores)
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    return {
        "avg_rec": (human_rec + machine_rec) / 2,
        "human_rec": human_rec,
        "machine_rec": machine_rec,
        "acc": 100.0 * accuracy_score(labels, preds),
        "precision": 100.0 * precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall": 100.0 * recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1": 100.0 * f1_score(labels, preds, pos_label=1, zero_division=0),
        "auroc": 100.0 * auc(fpr, tpr),
        "aupr": 100.0 * auc(recall_curve, precision_curve),
        "tpr_at_fpr_5": 100.0 * float(np.interp(0.05, fpr, tpr)),
        "fpr_at_tpr_95": 100.0 * float(np.interp(0.95, tpr, fpr)),
        "threshold": float(threshold),
    }
