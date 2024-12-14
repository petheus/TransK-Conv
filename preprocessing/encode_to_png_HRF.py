import cv2
import os
import tensorflow as tf
import numpy as np
from skimage.morphology import skeletonize
num_ = np.arange(45)
np.random.shuffle(num_)
#num_ = num_[:15]
num_ = ['01_dr.JPG', '02_dr.JPG', '03_dr.JPG', '04_dr.JPG', '05_dr.JPG',
        '01_g.jpg', '02_g.jpg', '03_g.jpg', '04_g.jpg', '05_g.jpg',
        '01_h.jpg', '02_h.jpg', '03_h.jpg', '04_h.jpg', '05_h.jpg',]
path = 'D:/paper_code/paper3hrf/HRF'

def label_weight(yy, m):
    y0 = yy[:, :, :1] + yy[:, :, 1:2]
    y0 = tf.cast(y0 == 1, dtype=tf.float32)
    y1 = yy[:, :, :1] + yy[:, :, 1:2] + yy[:, :, 2:]
    y1 = tf.cast(y1 > 0, dtype=tf.float32)
    y2 = yy[:, :, 2:] + yy[:, :, 1:2]
    y2 = tf.cast(y2 == 1, dtype=tf.float32)
    yy = tf.concat([y0, y1, y2], axis=-1)
    yy = yy.numpy()
    yy = np.expand_dims(yy, axis=0)
    m = np.expand_dims(m, axis=0)
    mmgai = np.ones(yy.shape, dtype='float32')
    yyyc = yy  # np.concatenate([yy[:,:,:,2:],yy[:,:,:,2:],yy[:,:,:,2:]],axis=-1)
    yyin = []
    for yyi in yyyc:
        yycom0 = skeletonize(yyi)
        yyin.append(yycom0)
    yyin = np.array(yyin)
    fifi = tf.ones([3, 3, 3, 1])

    yycom1 = tf.nn.depthwise_conv2d(yyyc, fifi, strides=[1, 1, 1, 1], padding='SAME').numpy()
    yycom2 = np.where(np.logical_or(np.logical_and(yycom1 < 9, yycom1 > 0, yyyc == 1), yyin > 0))

    yycom3 = np.where(np.logical_or(np.logical_and(yycom1 < 9, yycom1 > 0, yyyc == 0), yyyc > 0))
    mmgai[yycom3] = 2
    mmgai[yycom2] = 4
    mmmmg = m * mmgai
    mmmmg = np.array(mmmmg, dtype=np.uint8)
    return mmmmg[0, :, :, :]
import shutil
shutil.rmtree(path+'/test')
shutil.rmtree(path+'/training')
os.mkdir(path+'/test')
os.mkdir(path+'/training')
os.mkdir(path+'/test/mask')
os.mkdir(path+'/training/mask')
os.mkdir(path+'/test/images')
os.mkdir(path+'/training/images')
os.mkdir(path+'/test/av')
os.mkdir(path+'/training/av')
os.mkdir(path+'/training/weight_label')
for i,name in enumerate(os.listdir(path+'/images')):
    if name in num_:
        image = cv2.imread(path+'/images/'+name)[16:2320, 24:3480, :]
        for ix in range(4):
            for jy in range(6):
                cv2.imwrite(path+'/test/images/'+name.split('.')[0]+'{}{}.png'.format(ix,jy), image[ix*576:(ix+1)*576, jy*576:(jy+1)*576,:])
        image = cv2.imread(path + '/mask/' + name.split('.')[0]+'_mask.tif')[16:2320, 24:3480, :]
        for ix in range(4):
            for jy in range(6):
                cv2.imwrite(path + '/test/mask/' + name.split('.')[0] + '{}{}.png'.format(ix,jy), image[ix*576:(ix+1)*576, jy*576:(jy+1)*576,:])
        label = cv2.imread(path + '/HRF_AV_GT/' + name.split('.')[0] + '_AVmanual.png')# / 255
        label = cv2.resize(label, [3456, 2304],interpolation=cv2.INTER_NEAREST)
        for ix in range(4):
            for jy in range(6):
                cv2.imwrite(path + '/test/av/' + name.split('.')[0] + '{}{}.png'.format(ix,jy), label[ix*576:(ix+1)*576, jy*576:(jy+1)*576,:])
    else:
        image = cv2.imread(path+'/images/'+name)[16:2320, 24:3480, :]
        cv2.imwrite(path+'/training/images/'+name.split('.')[0]+'.png', image)
        image = cv2.imread(path + '/mask/' + name.split('.')[0]+'_mask.tif')[16:2320, 24:3480, :]
        cv2.imwrite(path + '/training/mask/' + name.split('.')[0] + '.png', image)
        label = cv2.imread(path + '/HRF_AV_GT/' + name.split('.')[0]+'_AVmanual.png')#/255
        label = cv2.resize(label,[3456, 2304],interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(path + '/training/av/' + name.split('.')[0] + '.png', label)
        label = label/255
        label = label_weight(label, image/255)
        cv2.imwrite(path + '/training/weight_label/' + name.split('.')[0] + '.png', label)