import numpy as np
import sklearn
import csv
from sklearn.metrics import roc_curve
from skimage.morphology import skeletonize, erosion
from sklearn.metrics import f1_score, accuracy_score
import skimage
import scipy
import matplotlib
matplotlib.use("Agg")          # 无图形界面服务器上也能保存图片
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score
)

def plot_auc_curve(y_true, y_pred):
    ytind = np.where((y_true[:,:,:1]+y_true[:,:,1:])>0.5)
    y_true = np.asarray(y_true[ytind]).ravel()
    y_pred = np.asarray(y_pred[ytind]).ravel()
    return roc_auc_score(y_true, y_pred)

def plot_pr_curve(y_true, y_pred, save_path="pr_curve.png", show=False):
    """
    计算并绘制 Precision-Recall 曲线，保存图片。

    参数
    ----
    y_true : array-like, shape (n_samples,)
        真实标签，0/1。
    y_pred : array-like, shape (n_samples,)
        预测分数（如正类概率 predict_proba(X)[:, 1]）。
        注意：若传入 0/1 硬标签，曲线只会有很少的折点，失去意义。
    save_path : str
        图片保存路径。
    show : bool
        是否在绘图后调用 plt.show()。

    返回
    ----
    dict: precision, recall, thresholds, ap, pr_auc
    """
    ytind = np.where((y_true[:,:,:1]+y_true[:,:,1:])>0.5)
    y_true = np.asarray(y_true[ytind]).ravel()
    y_pred = np.asarray(y_pred[ytind]).ravel()

    # ---------- 计算 PR 曲线 ----------
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    # AUC-PR：直接用梯形积分计算曲线下面积
    pr_auc = auc(recall, precision)

    # ---------- 绘制图像 ----------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="tab:blue", lw=2,
            label=f"PR curve AUC = {pr_auc:.3f})")
    ax.fill_between(recall, precision, alpha=0.15, color="tab:blue")

    # 基线：随机猜测时的 precision = 正类比例
    baseline = y_true.mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1,
               label=f"Baseline (pos. rate = {baseline:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

    fig.tight_layout()
    # savefig 必须在 show / close 之前调用，否则可能保存出空白图
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"PR 曲线已保存至: {save_path}")

    if show:
        plt.show()
    plt.close(fig)

    return pr_auc

def evaluation_code(prediction, groundtruth):
    '''
    Function to evaluate the performance of AV predictions with a given ground truth
    - prediction: should be an image array of [dim1, dim2, img_channels = 3] with arteries in red and veins in blue
    - groundtruth: same as above
    '''

    encoded_pred = np.zeros(prediction.shape[:2], dtype=int)
    encoded_gt = np.zeros(groundtruth.shape[:2], dtype=int)

    # convert white pixels to green pixels (which are ignored)
    white_ind = np.where(
        np.logical_and(groundtruth[:, :, 0] > 0, groundtruth[:, :, 1] > 0, groundtruth[:, :, 2] > 0))
    # if white_ind[0].size != 0:
    groundtruth[white_ind] = [0, 0, 255]

    # translate the images to arrays suited for sklearn metrics
    arteriole = np.where(np.logical_and(groundtruth[:, :, 0] > 0, groundtruth[:, :, 1] == 0))
    # arteriole = np.where(groundtruth[:, :, 0] > 0)
    encoded_gt[arteriole] = 1
    venule = np.where(np.logical_and(groundtruth[:, :, 1] > 0, groundtruth[:, :, 0] == 0))
    # venule = np.where(groundtruth[:, :, 1] > 0)
    encoded_gt[venule] = 2
    # arteriole = np.where(np.logical_and(prediction[:, :, 2] > 0.5, prediction[:, :, 0] > prediction[:, :, 1]))
    arteriole = np.where(prediction[:, :, 0] > 0.5)
    # arteriole = np.where(prediction[:, :, 0] > 0.5)
    encoded_pred[arteriole] = 1
    # venule = np.where(np.logical_and(prediction[:, :, 2] > 0.5, prediction[:, :, 0] < prediction[:, :, 1]))
    venule = np.where(prediction[:, :, 1] > 0.5)
    # venule = np.where(prediction[:, :, 1] > 0.5)
    encoded_pred[venule] = 2
    # retrieve the indices for the centerline pixels present in the prediction
    center = np.where(np.logical_and(
        np.logical_or(skeletonize(groundtruth[:, :, 0] > 0), skeletonize(groundtruth[:, :, 1] > 0)),
        encoded_pred[:, :] > 0))
    # print(set(list(encoded_pred.flatten())))
    encoded_pred_center = encoded_pred[center]
    encoded_gt_center = encoded_gt[center]

    # retrieve the indices for the centerline pixels present in the groundtruth
    center_comp = np.where(
        np.logical_or(skeletonize(groundtruth[:, :, 0] > 0), skeletonize(groundtruth[:, :, 1] > 0)))

    encoded_pred_center_comp = encoded_pred[center_comp]
    encoded_gt_center_comp = encoded_gt[center_comp]

    # retrieve the indices for discovered centerline pixels - limited to vessels wider than two pixels (for DRIVE)
    center_eroded = np.where(np.logical_and(
        np.logical_or(skeletonize(erosion(groundtruth[:, :, 0] > 0)), skeletonize(erosion(groundtruth[:, :, 1] > 0))),
        encoded_pred[:, :] > 0))

    encoded_pred_center_eroded = encoded_pred[center_eroded]
    encoded_gt_center_eroded = encoded_gt[center_eroded]

    # metrics over full image
    cur1_acc = accuracy_score(encoded_gt.flatten(), encoded_pred.flatten())
    cur1_F1 = f1_score(encoded_gt.flatten(), encoded_pred.flatten(), average='weighted')
    iny = np.logical_and(encoded_gt > 0, encoded_pred > 0)
    MCM = sklearn.metrics.confusion_matrix(encoded_gt[np.where(iny)].flatten(), encoded_pred[np.where(iny)].flatten())
    tn_sum = MCM[0, 0]
    fp_sum = MCM[0, 1]

    tp_sum = MCM[1, 1]
    fn_sum = MCM[1, 0]
    tnout = 1 * tn_sum
    fpout = 1 * fp_sum
    tpout = 1 * tp_sum
    fnout = 1 * fn_sum
    qu = 1  # (tp_sum+fn_sum)/(tp_sum+fn_sum+tn_sum+fp_sum)
    rec = np.sum(tp_sum / (tp_sum + fn_sum) * qu)
    pre = np.sum(tp_sum / (tp_sum + fp_sum) * qu)
    metrics1 = [np.sum(tp_sum / (tp_sum + fn_sum) * qu), np.sum(tn_sum / (tn_sum + fp_sum) * qu)
        , np.sum((tp_sum + tn_sum) / (tn_sum + fp_sum + tp_sum + fn_sum) * qu), 2 * rec * pre / (rec + pre)]

    # metrics over discovered centerline pixels
    cur2_acc = accuracy_score(encoded_gt_center.flatten(), encoded_pred_center.flatten())
    cur2_F1 = f1_score(encoded_gt_center.flatten(), encoded_pred_center.flatten(), average='weighted')
    tn_sum = np.count_nonzero(
        np.array(np.logical_and(encoded_gt_center.flatten() == 2, encoded_pred_center.flatten() == 2), dtype='int32'))
    fp_sum = np.count_nonzero(
        np.array(np.logical_and(encoded_gt_center.flatten() == 2, encoded_pred_center.flatten() == 1), dtype='int32'))
    tp_sum = np.count_nonzero(
        np.array(np.logical_and(encoded_gt_center.flatten() == 1, encoded_pred_center.flatten() == 1), dtype='int32'))
    fn_sum = np.count_nonzero(
        np.array(np.logical_and(encoded_gt_center.flatten() == 2, encoded_pred_center.flatten() == 1), dtype='int32'))
    rec = np.sum(tp_sum / (tp_sum + fn_sum) * qu)
    pre = np.sum(tp_sum / (tp_sum + fp_sum) * qu)
    metrics2 = [np.sum(tp_sum / (tp_sum + fn_sum) * qu), np.sum(tn_sum / (tn_sum + fp_sum) * qu)
        , np.sum((tp_sum + tn_sum) / (tn_sum + fp_sum + tp_sum + fn_sum) * qu), 2 * rec * pre / (rec + pre)]

    # metrics over discovered centerline pixels - limited to vessels wider than two pixels

    cur3_acc = accuracy_score(encoded_gt_center_eroded.flatten(), encoded_pred_center_eroded.flatten())
    cur3_F1 = f1_score(encoded_gt_center_eroded.flatten(), encoded_pred_center_eroded.flatten(), average='weighted')
    MCM = sklearn.metrics.multilabel_confusion_matrix(encoded_gt_center_eroded.flatten(),
                                                      encoded_pred_center_eroded.flatten(), labels=[1, 2])
    tn_sum = MCM[1:, 0, 0]
    fp_sum = MCM[1:, 0, 1]

    tp_sum = MCM[1:, 1, 1]
    fn_sum = MCM[1:, 1, 0]
    qu = 1  # (tp_sum+fn_sum)/(tp_sum+fn_sum+tn_sum+fp_sum)
    rec = np.sum(tp_sum / (tp_sum + fn_sum) * qu)
    pre = np.sum(tp_sum / (tp_sum + fp_sum) * qu)
    metrics3 = [np.sum(tp_sum / (tp_sum + fn_sum) * qu), np.sum(tn_sum / (tn_sum + fp_sum) * qu)
        , np.sum((tp_sum + tn_sum) / (tn_sum + fp_sum + tp_sum + fn_sum) * qu), 2 * rec * pre / (rec + pre)]

    # metrics over all centerline pixels in ground truth
    cur4_acc = accuracy_score(encoded_gt_center_comp.flatten(), encoded_pred_center_comp.flatten())
    cur4_F1 = f1_score(encoded_gt_center_comp.flatten(), encoded_pred_center_comp.flatten(), average='weighted')
    MCM = sklearn.metrics.multilabel_confusion_matrix(encoded_gt_center_comp.flatten(),
                                                      encoded_pred_center_comp.flatten(), labels=[0, 1, 2])
    tn_sum = MCM[1:, 0, 0]
    fp_sum = MCM[1:, 0, 1]

    tp_sum = MCM[1:, 1, 1]
    fn_sum = MCM[1:, 1, 0]
    qu = (tp_sum + fn_sum) / (tp_sum + fn_sum + tn_sum + fp_sum)
    rec = np.sum(tp_sum / (tp_sum + fn_sum) * qu)
    pre = np.sum(tp_sum / (tp_sum + fp_sum) * qu)
    metrics4 = [np.sum(tp_sum / (tp_sum + fn_sum) * qu), np.sum(tn_sum / (tn_sum + fp_sum) * qu)
        , np.sum((tp_sum + tn_sum) / (tn_sum + fp_sum + tp_sum + fn_sum) * qu), 2 * rec * pre / (rec + pre)]

    # finally, compute vessel detection rate
    vessel_ind = np.where(encoded_gt > 0)
    vessel_gt = encoded_gt[vessel_ind]
    vessel_pred = encoded_pred[vessel_ind]

    detection_rate = accuracy_score(vessel_gt.flatten(), vessel_pred.flatten())

    return metrics1, detection_rate, tnout, tpout, fpout, fnout


def binaryPostProcessing3(BinaryImage, removeArea, fillArea):
    """
    Post process the binary image
    :param BinaryImage:
    :param removeArea:
    :param fillArea:
    :return: Img_BW
    """

    BinaryImage[BinaryImage > 0] = 1

    ####takes 0.9s, result is good
    Img_BW = BinaryImage.copy()
    BinaryImage_Label = skimage.measure.label(Img_BW)
    for i, region in enumerate(skimage.measure.regionprops(BinaryImage_Label)):
        if region.area < removeArea:
            Img_BW[BinaryImage_Label == i + 1] = 0
        else:
            pass

    # ####takes 0.01s, result is bad
    # temptime = time.time()
    # Img_BW = morphology.remove_small_objects(BinaryImage, removeArea)
    # print "binaryPostProcessing3, ITK_LabelImage time:", time.time() - temptime

    Img_BW = skimage.morphology.closing(Img_BW, skimage.morphology.footprint_rectangle([3, 3]))
    # Img_BW = remove_small_holes(Img_BW, fillArea)

    Img_BW_filled = scipy.ndimage.binary_fill_holes(Img_BW)
    Img_BW_dif = np.uint8(Img_BW_filled) - np.uint8(Img_BW)
    Img_BW_difLabel = skimage.measure.label(Img_BW_dif)
    FilledImg = np.zeros(Img_BW.shape)
    for i, region in enumerate(skimage.measure.regionprops(Img_BW_difLabel)):
        if region.area < fillArea:
            FilledImg[Img_BW_difLabel == i + 1] = 1
        else:
            pass
        Img_BW[FilledImg > 0] = 1

    Img_BW = np.array(Img_BW, dtype=np.float32)
    return Img_BW


def softmax(inputs):
    x = np.exp(inputs)
    x = x / np.sum(x, axis=-1, keepdims=True)
    return x


def evaluatecal(filename=None):
    mask_2 = np.load('mask2.npy')
    y_p_2 = np.load('y_p2.npy')
    y_p_2 = y_p_2 * mask_2
    y_p_2 = np.concatenate([y_p_2[:, :, :, :, ::2], y_p_2[:, :, :, :, 1:2]], axis=-1)
    y_t_2 = np.load('y_t2.npy')
    y_t_2 = y_t_2 * mask_2
    y_t_2 = np.concatenate([y_t_2[:, :, :, :, ::2], y_t_2[:, :, :, :, 1:2]], axis=-1)

    y_ps = []
    y_ts = []
    metr_list = []
    for ii in range(len(y_p_2)):
        y_p = y_p_2[ii][0]
        ypmask = np.expand_dims(binaryPostProcessing3(y_p[:, :, 2] > 0.5, removeArea=100, fillArea=20), axis=-1)
        y_p[:, :, :2] = softmax(y_p[:, :, :2]) * ypmask
        y_p = y_p >= 0.5
        y_p0 = binaryPostProcessing3(y_p[:, :, 0], removeArea=100, fillArea=20)
        y_p1 = binaryPostProcessing3(y_p[:, :, 1], removeArea=100, fillArea=20)
        y_p2 = binaryPostProcessing3(y_p[:, :, 2], removeArea=100, fillArea=20)
        y_p = np.stack([y_p0, y_p1, y_p2], axis=-1)
        '''if if_result:
            cv2.imwrite('RITE/result/%d.png'%ii,np.array(y_p, dtype=np.uint8)*255)
            cv2.imwrite('RITE/result_true/%d.png'%ii,np.array(y_t_2[ii][0]*255, dtype=np.uint8))'''
        y_ps.append(1 * y_p)
        y_t = y_t_2[ii][0]
        y_ts.append(1 * y_t)

    y_ts = np.concatenate(y_ts, axis=0)
    y_ps = np.concatenate(y_ps, axis=0)
    metr, drt, tn_out, tp_out, fp_out, fn_out = evaluation_code(y_ps, np.array(y_ts, dtype='int32') * 255)
    roc_pr = plot_pr_curve(y_ts[:, :, :2], y_ps[:, :, :2], save_path='pr_'+filename+'.png')
    rocaucs=plot_auc_curve(y_t_2, y_p_2)
    metr.append(roc_pr)
    metr.append(rocaucs)
    #metr.insert(0, filename)
    #with open('ind1.csv', mode='a') as f:
        #csv_writer = csv.writer(f)
        #csv_writer.writerow(metr)
    #print('metrs', filename)
    #print(metr)
    return metr


if __name__ == '__main__':
    evaluatecal()