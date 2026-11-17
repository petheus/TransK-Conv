# per_image_metrics.py
import numpy as np
import skimage
import scipy.ndimage
import sklearn
from skimage.morphology import skeletonize, erosion
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                             auc, accuracy_score, f1_score,
                             confusion_matrix)


def binaryPostProcessing3(BinaryImage, removeArea, fillArea):
    BinaryImage = BinaryImage.copy()
    BinaryImage[BinaryImage > 0] = 1
    Img_BW = BinaryImage
    lab = skimage.measure.label(Img_BW)
    for i, region in enumerate(skimage.measure.regionprops(lab)):
        if region.area < removeArea:
            Img_BW[lab == i + 1] = 0
    Img_BW = skimage.morphology.closing(
        Img_BW, skimage.morphology.footprint_rectangle([3, 3]))
    filled = scipy.ndimage.binary_fill_holes(Img_BW)
    diff = np.uint8(filled) - np.uint8(Img_BW)
    dlab = skimage.measure.label(diff)
    F = np.zeros(Img_BW.shape)
    for i, region in enumerate(skimage.measure.regionprops(dlab)):
        if region.area < fillArea:
            F[dlab == i + 1] = 1
    Img_BW[F > 0] = 1
    return np.array(Img_BW, dtype=np.float32)


def _softmax(x, axis=-1):
    x = np.exp(x)
    x = x / np.sum(x, axis=axis, keepdims=True)
    return x


def _safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0


def _reorder_A_V_ves(arr):
    """getresults.py 中通道顺序是 [A, ves, V]；这里重排为 [A, V, ves]"""
    return np.concatenate([arr[..., ::2], arr[..., 1:2]], axis=-1)


def _postprocess_predictions(y_p):
    """与 evaluatecal 一致的后处理流程"""
    ypmask = np.expand_dims(binaryPostProcessing3(y_p[:, :, 2] > 0.5, removeArea=100, fillArea=20), axis=-1)
    y_p[:, :, :2] = _softmax(y_p[:, :, :2]) * ypmask
    y_p = y_p >= 0.5
    y_p0 = binaryPostProcessing3(y_p[:, :, 0], removeArea=100, fillArea=20)
    y_p1 = binaryPostProcessing3(y_p[:, :, 1], removeArea=100, fillArea=20)
    y_p2 = binaryPostProcessing3(y_p[:, :, 2], removeArea=100, fillArea=20)
    y_p = np.stack([y_p0, y_p1, y_p2], axis=-1)
    return y_p


def compute_single_image_metrics(y_p_raw, y_t_raw, mask):
    """
    y_p_raw, y_t_raw, mask : (H, W, 3)，通道顺序 [A, ves, V]
    返回：dict of scalar（与论文指标一致 + 更标准的逐通道 AUC）
    """
    y_p = _reorder_A_V_ves(np.asarray(y_p_raw, dtype=np.float32).copy())
    y_t = _reorder_A_V_ves(np.asarray(y_t_raw, dtype=np.float32).copy())
    mask = np.asarray(mask, dtype=np.float32).copy()
    y_p = y_p * mask
    y_t = y_t * mask

    out = {}
    names = ['A', 'V', 'Ves']

    # ---- 逐通道二值指标（阈值 0.5，未后处理） ----
    for c, name in enumerate(names):
        p = (y_p[..., c] > 0.5).astype(np.uint8)
        t = (y_t[..., c] > 0.5).astype(np.uint8)
        tp = int(np.sum((p == 1) & (t == 1)))
        fp = int(np.sum((p == 1) & (t == 0)))
        fn = int(np.sum((p == 0) & (t == 1)))
        tn = int(np.sum((p == 0) & (t == 0)))
        out[f'dice_{name}'] = _safe_div(2 * tp, 2 * tp + fp + fn)
        out[f'acc_{name}'] = _safe_div(tp + tn, tp + tn + fp + fn)
        out[f'sens_{name}'] = _safe_div(tp, tp + fn)
        out[f'spec_{name}'] = _safe_div(tn, tn + fp)
        out[f'f1_{name}'] = _safe_div(2 * tp, 2 * tp + fp + fn)

        # raw-score AUC-ROC / AUC-PR
        if 0 < t.sum() < t.size:
            try:
                out[f'aucroc_{name}'] = float(roc_auc_score(
                    t.ravel(), y_p[..., c].ravel()))
                prec, rec, _ = precision_recall_curve(
                    t.ravel(), y_p[..., c].ravel())
                out[f'aucpr_{name}'] = float(auc(rec, prec))
            except Exception:
                out[f'aucroc_{name}'] = np.nan
                out[f'aucpr_{name}'] = np.nan
        else:
            out[f'aucroc_{name}'] = np.nan
            out[f'aucpr_{name}'] = np.nan

    # ---- A/V 分类 AUC（仅在血管像素上） ----
    ves = (y_t[..., 0] > 0.5) | (y_t[..., 1] > 0.5)
    if ves.any():
        t_av = (y_t[..., 0][ves] > 0.5).astype(np.int32)
        p_av = y_p[..., 0][ves] / (y_p[..., 0][ves] + y_p[..., 1][ves] + 1e-8)
        if 0 < t_av.sum() < len(t_av):
            try:
                out['aucroc_AV'] = float(roc_auc_score(t_av, p_av))
                prec, rec, _ = precision_recall_curve(t_av, p_av)
                out['aucpr_AV'] = float(auc(rec, prec))
            except Exception:
                out['aucroc_AV'] = np.nan
                out['aucpr_AV'] = np.nan
        else:
            out['aucroc_AV'] = np.nan
            out['aucpr_AV'] = np.nan
    else:
        out['aucroc_AV'] = np.nan
        out['aucpr_AV'] = np.nan

    # ---- 复现 evaluatecal 中 metrics1 与 DR 的后处理流程 ----
    y_p_post = _postprocess_predictions(y_p)
    y_t = np.reshape(y_t,y_p_post.shape)

    encoded_pred = np.zeros(y_p_post.shape[:2], dtype=int)
    encoded_gt = np.zeros(y_t.shape[:2], dtype=int)

    # convert white pixels to green pixels (which are ignored)
    white_ind = np.where(
        np.logical_and(y_t[:, :, 0] > 0, y_t[:, :, 1] > 0, y_t[:, :, 2] > 0))
    # if white_ind[0].size != 0:
    y_t[white_ind] = [0, 0, 255]

    # translate the images to arrays suited for sklearn metrics
    arteriole = np.where(np.logical_and(y_t[:, :, 0] > 0, y_t[:, :, 1] == 0))
    # arteriole = np.where(y_t[:, :, 0] > 0)
    encoded_gt[arteriole] = 1
    venule = np.where(np.logical_and(y_t[:, :, 1] > 0, y_t[:, :, 0] == 0))
    # venule = np.where(y_t[:, :, 1] > 0)
    encoded_gt[venule] = 2
    arteriole = np.where(y_p_post[:, :, 0] > 0.5)
    encoded_pred[arteriole] = 1
    venule = np.where(y_p_post[:, :, 1] > 0.5)
    encoded_pred[venule] = 2
    # retrieve the indices for the centerline pixels present in the prediction
    # metrics over full image
    cur1_acc = accuracy_score(encoded_gt.flatten(), encoded_pred.flatten())
    cur1_F1 = f1_score(encoded_gt.flatten(), encoded_pred.flatten(), average='weighted')
    iny = np.logical_and(encoded_gt > 0, encoded_pred > 0)
    MCM = sklearn.metrics.confusion_matrix(encoded_gt[np.where(iny)].flatten(), encoded_pred[np.where(iny)].flatten())
    tn_sum = MCM[0, 0]
    fp_sum = MCM[0, 1]

    tp_sum = MCM[1, 1]
    fn_sum = MCM[1, 0]


    rec = _safe_div(tp_sum, tp_sum + fn_sum)
    pre = _safe_div(tp_sum, tp_sum + fp_sum)
    out['metrics1_Se'] = rec
    out['metrics1_Sp'] = _safe_div(tn_sum, tn_sum + fp_sum)
    out['metrics1_Acc'] = _safe_div(
        tp_sum + tn_sum, tp_sum + tn_sum + fp_sum + fn_sum)
    out['metrics1_F1'] = 2 * rec * pre / (rec + pre) if (rec + pre) > 0 else 0.0

    vessel_ind = np.where(encoded_gt > 0)
    if len(vessel_ind[0]) > 0:
        out['DR'] = float(np.mean(encoded_gt[vessel_ind] == encoded_pred[vessel_ind]))
    else:
        out['DR'] = np.nan

    return out


def compute_per_image_metrics(yps, ys, ms):
    """
    yps, ys, ms : 列表，每个元素 shape (1, H, W, 3)，通道顺序 [A, ves, V]
    返回 dict: {metric_name: np.array(N,)}，另含 'image_index'
    """
    n = len(yps)
    rows = []
    for i in range(n):
        rows.append(compute_single_image_metrics(
            np.asarray(yps[i])[0], np.asarray(ys[i])[0], np.asarray(ms[i])[0]))
    keys = list(rows[0].keys())
    out = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys}
    out['image_index'] = np.arange(n)
    print(out)
    return out