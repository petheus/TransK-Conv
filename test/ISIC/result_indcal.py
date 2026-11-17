# result_indcal.py  —— 只列出 evaluatecal 及其依赖，其余函数（plot_pr_curve 等）不动

import os
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm

from per_image_metrics_isic import compute_single_image_metrics


# ===== 1) plot_pr_curve：在内部做防御性二值化 =====
def plot_pr_curve(y_true, y_pred, save_path=None, show=False):
    """
    与原来一致，只是把 y_true 强制转为 int 0/1，
    兼容 0/255、0/1、float 0/1 各种输入。
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel().astype(np.float64)

    # 兼容 >1 的取值（如 0/255），否则按 0.5 阈值二值化
    if y_true.max() > 1.0:
        y_true = (y_true > 127.5).astype(np.uint8)
    else:
        y_true = (y_true > 0.5).astype(np.uint8)

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    return pr_auc


# ===== 2) evaluatecal：读取后强制二值化 =====
def evaluatecal(if_result=False, filename=None,
                result_dir='./result', per_image_dir=None):

    result_paths = [p for p in os.listdir(result_dir) if p.endswith('.npy')]
    result_paths.sort()
    if not result_paths:
        raise RuntimeError('empty dir: %s' % result_dir)

    MCM = 0.
    MCMs = []
    roc_prs = []
    rocaucs = []
    per_img_metrics = []
    image_names = []

    for result_name in tqdm(result_paths):
        resulty = np.load(os.path.join(result_dir, result_name))

        # ★ 关键修复：强制为 uint8（sklearn 新版要求 y_true 是整型二值）
        y_p_prob = resulty[0].astype(np.float64)              # 概率
        y_t_true = (resulty[1] > 0.5).astype(np.uint8)        # 硬标签，int
        y_p_bin  = (y_p_prob > 0.5).astype(np.uint8)

        # 原版逻辑：MCM 累积（不变）
        MCMi = sklearn.metrics.confusion_matrix(
            y_t_true, y_p_bin, labels=[0, 1]).astype(np.float64)
        MCMs.append(MCMi.copy())
        MCM += MCMi

        # 原版逻辑：PR / AUC（不变，但 y_t_true 已是 uint8，不会再报错）
        pr_auc = plot_pr_curve(y_t_true, y_p_prob)
        roc_prs.append(float(pr_auc))
        rocaucs.append(float(roc_auc_score(y_t_true, y_p_prob)))

        # 逐图像指标（流式，算完即丢）
        per_img_metrics.append(
            compute_single_image_metrics(y_p_prob, y_t_true))
        image_names.append(result_name[:-4])

    # 逐图像指标落盘（不变）
    if per_image_dir is not None:
        os.makedirs(per_image_dir, exist_ok=True)
        keys = list(per_img_metrics[0].keys())
        out = {k: np.array([r[k] for r in per_img_metrics], dtype=np.float64)
               for k in keys}
        out['image_name'] = np.array(image_names)
        np.savez(os.path.join(per_image_dir, 'metrics.npz'), **out)

    if if_result:
        np.save('ind' + filename + '.npy', np.array(MCMs))

    # 原版 4 指标 + 2 AUC（不变）
    tn_sum = MCM[0, 0]; fp_sum = MCM[0, 1]
    tp_sum = MCM[1, 1]; fn_sum = MCM[1, 0]
    qu = 1
    rec = np.sum(tp_sum / (tp_sum + fn_sum) * qu)
    pre = np.sum(tp_sum / (tp_sum + fp_sum) * qu)
    metrone = [
        tp_sum / (tp_sum + fn_sum + fp_sum),
        np.sum((tp_sum + tn_sum) / (tn_sum + fp_sum + tp_sum + fn_sum) * qu),
        pre, rec,
    ]
    metr = metrone
    metr.append(np.average(roc_prs))
    metr.append(np.average(rocaucs))
    metr = np.array(metr)
    print('metrs', filename)
    print(metr)
    return metr
if __name__ == '__main__':
    evaluatecal()