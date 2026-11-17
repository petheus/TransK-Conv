import numpy as np
import sklearn
import csv
from sklearn.metrics import roc_curve, auc
from skimage.morphology import skeletonize, erosion, dilation
from sklearn.metrics import f1_score, accuracy_score
from skimage.morphology import square
import skimage
import scipy
import cv2
import os

def evaluatecal(if_result=False, filename=None):
    result_paths = os.listdir('./result')
    result_paths = list(filter(lambda x: x[-3:]=='npy', result_paths))
    #metr = float(0)
    imnum = 0
    MCM = 0.
    MCMs = []
    for result_name in result_paths:
        resulty = np.load('./result/'+result_name)
        y_p_2 = resulty[0]
        y_t_2 = resulty[1]
        MCMi = sklearn.metrics.confusion_matrix(y_t_2, y_p_2, labels=[0,1])
        MCMi = MCMi.astype(np.float64)
        MCMs.append(1.*MCMi)
        MCM += MCMi
    if if_result:
        np.save('ind'+filename+'.npy', MCMs)
    tn_sum = MCM[0, 0]
    fp_sum = MCM[0, 1]

    tp_sum = MCM[1, 1]
    fn_sum = MCM[1, 0]
    #tnout = 1 * tn_sum
    #fpout = 1 * fp_sum
    #tpout = 1 * tp_sum
    #fnout = 1 * fn_sum
    qu = 1  # (tp_sum+fn_sum)/(tp_sum+fn_sum+tn_sum+fp_sum)
    rec = np.sum(tp_sum / (tp_sum + fn_sum) * qu)
    pre = np.sum(tp_sum / (tp_sum + fp_sum) * qu)
    metrone = [np.sum((tp_sum + tn_sum) / (tn_sum + fp_sum + tp_sum + fn_sum) * qu), 2 * rec * pre / (rec + pre),
                np.sum(tn_sum / (tn_sum + fp_sum) * qu), np.sum(tp_sum / (tp_sum + fn_sum) * qu), pre, 2*tp_sum/(2*tp_sum+fn_sum+fp_sum), tp_sum/(tp_sum+fn_sum+fp_sum)]
    metr = np.array(metrone)
    #imnum += 1 



    print(metr)
    return metr


if __name__ == '__main__':
    evaluatecal()
