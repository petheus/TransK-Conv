import cv2
import os
import tensorflow as tf
import numpy as np
from skimage.morphology import skeletonize

path = './RITE'
#os.mkdir(path+'/test/mask_png')
#os.mkdir(path+'/training/mask_png')
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
for name in os.listdir(path+'/test/images'):
    image = cv2.imread(path+'/test/images/'+name)
    cv2.imwrite(path+'/test/images_png/'+name[:-4]+'.png', image)
    image = cv2.imread(path + '/test/mask/' + name[:-4]+'_mask.png')
    #cv2.imwrite(path + '/test/mask_png/' + name[:-4] + '.png', image)

for name in os.listdir(path+'/training/images'):
    image = cv2.imread(path+'/training/images/'+name)
    #cv2.imwrite(path+'/training/images_png/'+name[:-4]+'.png', image)
    image = cv2.imread(path + '/training/mask/' + name[:-4]+'_mask.jpg')
    #cv2.imwrite(path + '/training/mask_png/' + name[:-4] + '.png', image)
    label = cv2.imread(path + '/training/av/' + name[:-4]+'.png')/255
    label = label_weight(label, image/255)
    cv2.imwrite(path + '/training/weight_label/' + name[:-4] + '.png', label)