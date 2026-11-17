import os
import numpy as np
import sklearn
import csv
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc)
from medpy import metric
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm


# ---------- 安全版 PR / ROC（缺类返回 nan，不抛异常） ----------
def _safe_pr_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.uint8).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if len(np.unique(y_true)) < 2:
        return np.nan
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def _safe_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.uint8).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


# ---------- 保留你原版 ----------
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


# ---------- 新版 evaluatecal ----------
def evaluatecal(if_result=False, filename=None,
                result_dir='./result', trueresult_dir='./trueresult',
                per_image_dir=None):
    """
    返回 [dice_RV, dice_Myo, dice_LV, dice_mean, pr_mean, roc_mean]
    —— 与旧 evaluatecal 返回值 100% 等价（仅去掉了无用画图）。
    额外：如果给了 per_image_dir，就写出 metrics.npz 供后续 Wilcoxon 用。
    """
    result_paths = [p for p in os.listdir(result_dir) if p.endswith('.npy')]
    result_paths.sort()
    if not result_paths:
        raise RuntimeError('empty dir: ' + result_dir)

    metr = []          # list of [(dice_i, hd95_i) for i in 1..3]
    pr_scores = []
    roc_scores = []
    per_img_rows = []

    for result_name in tqdm(result_paths):
        y_p_2 = np.load(os.path.join(result_dir, result_name))          # (D,4,H,W)
        y_t_2 = np.load(os.path.join(trueresult_dir, result_name))      # (D*H*W,)

        # ===== 替换 OneHotEncoder：直接按类 == i 索引，缺类不会崩 =====
        prs_per_class  = []
        rocs_per_class = []
        for i in range(1, 4):
            y_t_i = (y_t_2 == i)
            y_s_i = y_p_2[:, i, :, :].reshape(-1)
            prs_per_class.append(_safe_pr_auc(y_t_i, y_s_i))
            rocs_per_class.append(_safe_roc_auc(y_t_i, y_s_i))
        pr_scores.append(float(np.nanmean(prs_per_class)))
        roc_scores.append(float(np.nanmean(rocs_per_class)))

        # ===== Dice / HD95（保留你原逻辑与 0 值处理） =====
        y_p_argmax = np.reshape(np.argmax(y_p_2, axis=1).astype(np.uint8), [-1])
        dice_per_class  = []
        hd95_per_class  = []
        for i in range(1, 4):
            d, h = calculate_metric_percase(y_p_argmax == i, y_t_2 == i)
            dice_per_class.append(float(d))
            hd95_per_class.append(float(h))
        metr.append(list(zip(dice_per_class, hd95_per_class)))

        # ===== 逐 volume 收集（流式，不驻留大数组） =====
        per_img_rows.append({
            'name':     result_name[:-4],
            'dice_RV':  dice_per_class[0],
            'dice_Myo': dice_per_class[1],
            'dice_LV':  dice_per_class[2],
            'hd95_RV':  hd95_per_class[0],
            'hd95_Myo': hd95_per_class[1],
            'hd95_LV':  hd95_per_class[2],
            'pr_RV':    prs_per_class[0],
            'pr_Myo':   prs_per_class[1],
            'pr_LV':    prs_per_class[2],
            'roc_RV':   rocs_per_class[0],
            'roc_Myo':  rocs_per_class[1],
            'roc_LV':   rocs_per_class[2],
        })

    # ===== 逐 volume 指标落盘（每 fold 一个 npz，≈ 5 KB） =====
    if per_image_dir is not None:
        os.makedirs(per_image_dir, exist_ok=True)
        out = {}
        for k in per_img_rows[0]:
            if k == 'name':
                out['name'] = np.array([r['name'] for r in per_img_rows])
            else:
                out[k] = np.array([r[k] for r in per_img_rows], dtype=np.float64)
        for prefix in ['dice', 'hd95', 'pr', 'roc']:
            out[prefix + '_mean'] = np.nanmean(
                np.stack([out[prefix + '_RV'],
                          out[prefix + '_Myo'],
                          out[prefix + '_LV']], axis=0), axis=0)
        np.savez(os.path.join(per_image_dir, 'metrics.npz'), **out)

    # ===== 与旧 evaluatecal 完全等价的汇总返回 =====
    metr1 = np.array(metr)[:, :, 0]                    # (N, 3)
    if if_result:
        np.save('ind' + filename + '.npy',
                np.mean(metr1, 1).astype(np.float32))
    metr  = np.mean(metr, 0).astype(np.float32)        # (3, 2)
    metr0 = metr[:, 0]                                 # (3,) dice
    print('metrs', filename)
    print([metr0[0], metr0[1], metr0[2], np.mean(metr0),
           np.mean(pr_scores), np.mean(roc_scores)])
    return [float(metr0[0]), float(metr0[1]), float(metr0[2]),
            float(np.mean(metr0)),
            float(np.mean(pr_scores)), float(np.mean(roc_scores))]