from scipy import ndimage
import skimage.morphology as sm
import cv2
import numpy as np
import os
path = './RITE'
trainimages = os.listdir(path + '/training/images_png')
trainimages = np.array(list(trainimages))
train_labelnames = [path + '/training/av/' + name for name in trainimages]
for labelname in train_labelnames:
    label = cv2.imread(labelname)/255
    y0 = label[:, :, 0] + label[:, :, 1]
    y0 = np.array(y0==1, dtype=np.uint8)
    y1 = label[:, :, 0] + label[:, :, 1] + label[:, :, 2]
    y1 = np.array(y1 > 0, dtype=np.uint8)
    y2 = label[:, :, 2] + label[:, :, 1]
    y2 = np.array(y2 == 1, dtype=np.uint8)
    LabelArtery_dil0 = sm.dilation(y0,sm.square(13))*255
    LabelArtery_dil1 = sm.dilation(y1, sm.square(13)) * 255
    LabelArtery_dil2 = sm.dilation(y2, sm.square(13)) * 255
    LabelArtery_dil = np.stack([LabelArtery_dil0, LabelArtery_dil1, LabelArtery_dil2], axis=-1)
    print(np.sum(np.array(LabelArtery_dil!=0,dtype=np.int32))==np.sum(np.array(LabelArtery_dil==255,dtype=np.int32)))
    cv2.imwrite('dil/13_'+labelname.split('/')[-1], LabelArtery_dil)

for labelname in train_labelnames:
    label = cv2.imread(labelname)/255
    y0 = label[:, :, 0] + label[:, :, 1]
    y0 = np.array(y0==1, dtype=np.uint8)
    y1 = label[:, :, 0] + label[:, :, 1] + label[:, :, 2]
    y1 = np.array(y1 > 0, dtype=np.uint8)
    y2 = label[:, :, 2] + label[:, :, 1]
    y2 = np.array(y2 == 1, dtype=np.uint8)
    LabelArtery_dil0 = sm.dilation(y0,sm.square(9))*255
    LabelArtery_dil1 = sm.dilation(y1, sm.square(9)) * 255
    LabelArtery_dil2 = sm.dilation(y2, sm.square(9)) * 255
    LabelArtery_dil = np.stack([LabelArtery_dil0, LabelArtery_dil1, LabelArtery_dil2], axis=-1)
    print(np.sum(np.array(LabelArtery_dil!=0,dtype=np.int32))==np.sum(np.array(LabelArtery_dil==255,dtype=np.int32)))
    cv2.imwrite('dil/9_'+labelname.split('/')[-1], LabelArtery_dil)

for labelname in train_labelnames:
    label = cv2.imread(labelname)/255
    y0 = label[:, :, 0] + label[:, :, 1]
    y0 = np.array(y0==1, dtype=np.uint8)
    y1 = label[:, :, 0] + label[:, :, 1] + label[:, :, 2]
    y1 = np.array(y1 > 0, dtype=np.uint8)
    y2 = label[:, :, 2] + label[:, :, 1]
    y2 = np.array(y2 == 1, dtype=np.uint8)
    LabelArtery_dil0 = sm.dilation(y0,sm.square(5))*255
    LabelArtery_dil1 = sm.dilation(y1, sm.square(5)) * 255
    LabelArtery_dil2 = sm.dilation(y2, sm.square(5)) * 255
    LabelArtery_dil = np.stack([LabelArtery_dil0, LabelArtery_dil1, LabelArtery_dil2], axis=-1)
    print(np.sum(np.array(LabelArtery_dil!=0,dtype=np.int32))==np.sum(np.array(LabelArtery_dil==255,dtype=np.int32)))
    cv2.imwrite('dil/5_'+labelname.split('/')[-1], LabelArtery_dil)