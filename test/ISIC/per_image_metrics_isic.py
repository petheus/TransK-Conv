# per_image_metrics_isic.py
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def _safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0


def compute_single_image_metrics(y_p, y_t):
    """y_p 概率（任意 shape，内部 ravel）；y_t 硬标签 {0,1}"""
    p = np.asarray(y_p, dtype=np.float64).ravel()
    t = (np.asarray(y_t, dtype=np.float64).ravel() > 0.5).astype(np.uint8)
    pb = (p > 0.5).astype(np.uint8)

    tp = int(np.sum((pb == 1) & (t == 1)))
    fp = int(np.sum((pb == 1) & (t == 0)))
    fn = int(np.sum((pb == 0) & (t == 1)))
    tn = int(np.sum((pb == 0) & (t == 0)))

    out = {
        'IoU': _safe_div(tp, tp + fp + fn),
        'Acc': _safe_div(tp + tn, tp + tn + fp + fn),
        'Sen': _safe_div(tp, tp + fn),
        'Spe': _safe_div(tn, tn + fp),
        'Pre': _safe_div(tp, tp + fp),
        'F1':  _safe_div(2 * tp, 2 * tp + fp + fn),
    }
    out['Dice'] = out['F1']

    if 0 < t.sum() < t.size:
        try:
            out['AUC_ROC'] = float(roc_auc_score(t, p))
            prec, rec, _ = precision_recall_curve(t, p)
            out['AUC_PR'] = float(auc(rec, prec))
        except Exception:
            out['AUC_ROC'] = np.nan
            out['AUC_PR'] = np.nan
    else:
        out['AUC_ROC'] = np.nan
        out['AUC_PR'] = np.nan
    return out