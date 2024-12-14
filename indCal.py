import numpy as np
import sklearn
import csv
from sklearn.metrics import roc_curve, auc
from skimage.morphology import skeletonize, erosion, dilation
from sklearn.metrics import f1_score, accuracy_score
from skimage.morphology import square
import skimage
import scipy


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
    #arteriole = np.where(groundtruth[:, :, 0] > 0)
    encoded_gt[arteriole] = 1
    venule = np.where(np.logical_and(groundtruth[:, :, 1] > 0, groundtruth[:, :, 0] == 0))
    #venule = np.where(groundtruth[:, :, 1] > 0)
    encoded_gt[venule] = 2
    #arteriole = np.where(np.logical_and(prediction[:, :, 2] > 0.5, prediction[:, :, 0] > prediction[:, :, 1]))
    arteriole = np.where(prediction[:, :, 0]>0.5)
    # arteriole = np.where(prediction[:, :, 0] > 0.5)
    encoded_pred[arteriole] = 1
    #venule = np.where(np.logical_and(prediction[:, :, 2] > 0.5, prediction[:, :, 0] < prediction[:, :, 1]))
    venule = np.where(prediction[:, :, 1]>0.5)
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
    metrics1 = [np.sum((tp_sum + tn_sum) / (tn_sum + fp_sum + tp_sum + fn_sum) * qu), 2 * rec * pre / (rec + pre),
                np.sum(tn_sum / (tn_sum + fp_sum) * qu), np.sum(tp_sum / (tp_sum + fn_sum) * qu)]

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
    metrics2 = [np.sum((tp_sum + tn_sum) / (tn_sum + fp_sum + tp_sum + fn_sum) * qu), 2 * rec * pre / (rec + pre),
                np.sum(tn_sum / (tn_sum + fp_sum) * qu), np.sum(tp_sum / (tp_sum + fn_sum) * qu)]

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
    metrics3 = [np.sum((tp_sum + tn_sum) / (tn_sum + fp_sum + tp_sum + fn_sum) * qu), 2 * rec * pre / (rec + pre),
                np.sum(tn_sum / (tn_sum + fp_sum) * qu), np.sum(tp_sum / (tp_sum + fn_sum) * qu)]

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
    metrics4 = [np.sum((tp_sum + tn_sum) / (tn_sum + fp_sum + tp_sum + fn_sum) * qu), 2 * rec * pre / (rec + pre),
                np.sum(tn_sum / (tn_sum + fp_sum) * qu), np.sum(tp_sum / (tp_sum + fn_sum) * qu)]

    # finally, compute vessel detection rate
    vessel_ind = np.where(encoded_gt > 0)
    vessel_gt = encoded_gt[vessel_ind]
    vessel_pred = encoded_pred[vessel_ind]

    detection_rate = accuracy_score(vessel_gt.flatten(), vessel_pred.flatten())

    return [metrics1, metrics2, metrics3, metrics4], detection_rate, tnout, tpout, fpout, fnout


def evaluation_code2(prediction, groundtruth):
    prediction = np.reshape(prediction, [-1, 3])
    groundtruth = np.reshape(groundtruth, [-1, 3])
    MCM = sklearn.metrics.multilabel_confusion_matrix(groundtruth, prediction)
    tn_sum = MCM[:2, 0, 0]
    fp_sum = MCM[:2, 0, 1]

    tp_sum = MCM[:2, 1, 1]
    fn_sum = MCM[:2, 1, 0]
    rec = np.average(tp_sum / (tp_sum + fn_sum))
    pre = np.average(tp_sum / (tp_sum + fp_sum))
    metrics1 = [np.average((tp_sum + tn_sum) / (tn_sum + fp_sum + tp_sum + fn_sum)), 2 * rec * pre / (rec + pre),
                np.average(tn_sum / (tn_sum + fp_sum)), np.average(tp_sum / (tp_sum + fn_sum))]
    return [metrics1]

def binaryPostProcessing3(BinaryImage, removeArea, fillArea):
    """
    Post process the binary image
    :param BinaryImage:
    :param removeArea:
    :param fillArea:
    :return: Img_BW
    """

    BinaryImage[BinaryImage>0]=1

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


    Img_BW = skimage.morphology.binary_closing(Img_BW, skimage.morphology.square(3))
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
    x= np.exp(inputs)
    x= x/np.sum(x,axis=-1,keepdims=True)
    return x

def evaluatecal():
    mask_2 = np.load('mask2.npy')
    y_p_2 = np.load('y_p2.npy')
    y_p_2 = y_p_2 * mask_2
    y_p_2 = np.concatenate([y_p_2[:, :, :, :, ::2], y_p_2[:, :, :, :, 1:2]], axis=-1)
    y_t_2 = np.load('y_t2.npy')
    y_t_2 = y_t_2 * mask_2
    y_t_2 = np.concatenate([y_t_2[:, :, :, :, ::2], y_t_2[:, :, :, :, 1:2]], axis=-1)
    
    
    
    metrs = np.array([[0., 0., 0., 0.] for ii in range(4)])
    drts = 0
    tnn = 0
    tpp = 0
    fpp = 0
    fnn = 0
    y_ps=[]
    y_ts=[]
    yms =[]
    for ii in range(len(y_p_2)):
        y_p = y_p_2[ii][0]
        ypmask = np.expand_dims(binaryPostProcessing3(y_p[:,:,2]>0.5, removeArea=100, fillArea=20),axis=-1)
        y_p[:,:,:2] = softmax(y_p[:,:,:2])*ypmask
        y_p = y_p >=0.5
        y_p0 = binaryPostProcessing3(y_p[:,:,0], removeArea=100, fillArea=20)
        y_p1 = binaryPostProcessing3(y_p[:,:,1], removeArea=100, fillArea=20)
        y_p2 = binaryPostProcessing3(y_p[:,:,2], removeArea=100, fillArea=20)
        y_p = np.stack([y_p0,y_p1,y_p2],axis=-1)
        y_ps.append(1*y_p)
        y_t = y_t_2[ii][0]
        y_ts.append(1*y_t)
        metr, drt, tn_out, tp_out, fp_out, fn_out = evaluation_code(y_p, np.array(y_t, dtype='int32') * 255)
        metr = np.array(metr)
        metrs = metrs + metr
        with open('ind.csv', mode='w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(metr.flatten())

    print('metrs')
    print(metrs / len(y_p_2))
    y_ts=np.concatenate(y_ts,axis=0)
    y_ps=np.concatenate(y_ps,axis=0)
    metr, drt, tn_out, tp_out, fp_out, fn_out = evaluation_code(y_ps, np.array(y_ts, dtype='int32') * 255)
    metr = np.array(metr)
    print('metrs')
    print(metr)
    with open('ind2.csv', mode='w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(metr.flatten())


if __name__ == '__main__':
    evaluatecal()
