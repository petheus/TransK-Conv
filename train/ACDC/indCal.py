import numpy as np
import sklearn
import csv
from medpy import metric
import cv2
import os

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0

def evaluatecal(if_result=False, filename=None):
    result_paths = os.listdir('./result')
    result_paths = list(filter(lambda x: x[-3:]=='npy', result_paths))
    metr = []
    for result_name in result_paths:
        resulty = np.load('./result/'+result_name)
        y_p_2 = resulty[0]
        y_t_2 = resulty[1]
        metric_list = []
        for i in range(1, 4):
            metric_list.append(calculate_metric_percase(
                y_p_2 == i, y_t_2 == i))
        metr.append(metric_list)
    metr1 = np.array(metr)[:,:,0]
    if if_result:
        np.save('ind'+filename+'.npy', np.mean(metr1, 1).astype(np.float32))
    metr = np.mean(metr, 0).astype(np.float32)
    metr0 = metr[:,0]
    if if_result:
        with open('ind'+filename+'.csv', mode='w+') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([metr0[0], metr0[1], metr0[2], np.mean(metr0)])
    return metr


if __name__ == '__main__':
    evaluatecal()
