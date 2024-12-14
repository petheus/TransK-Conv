import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpunum = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpunum
import tensorflow as tf
from net.netsUnets import ConnectUnet
import albumentations as A
import numpy as np
import cv2
import sys
import time
from tqdm import tqdm
from indCal import evaluatecal
#sys.stdout = open('output.txt', 'w')
UNETN = 3
save_path = './hrfsave/'+gpunum
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
path = './HRF'
np.random.seed(10)
num_ = np.arange(20)
np.random.shuffle(num_)
num_ = num_
trainimages = os.listdir(path + '/training/images')
trainidx = list(map(lambda x: x.split('.')[-1]=='png', trainimages))
trainimages = np.array(list(trainimages))
trainimages = trainimages[np.where(trainidx)]
trainimages = trainimages[num_]
testimages = os.listdir(path + '/test/images')
testidx = list(map(lambda x: x.split('.')[-1]=='png', testimages))
testimages = np.array(list(testimages))
testimages = testimages[np.where(testidx)].tolist()
train_imagenames = tf.constant([path + '/training/images/' + name for name in trainimages])
test_imagenames = tf.constant([path + '/test/images/' + name for name in testimages])
train_labelnames = tf.constant([path + '/training/av/' + name for name in trainimages])

test_labelnames = tf.constant([path + '/test/av/' + name for name in testimages])
train_masknames = tf.constant([path + '/training/mask/' + name for name in trainimages])
test_masknames = tf.constant([path + '/test/mask/' + name for name in testimages])
train_wmasknames = tf.constant([path + '/training/weight_label/' + name for name in trainimages])
train_dataset = tf.data.Dataset.from_tensor_slices((train_imagenames, train_labelnames, train_masknames, train_wmasknames))
test_dataset = tf.data.Dataset.from_tensor_slices((test_imagenames, test_labelnames, test_masknames))


def _decode_and_resize(imagenames, labelnames, masknames, wmasknames):
    image_string = tf.io.read_file(imagenames)
    image_resized = tf.image.decode_png(image_string)
    #image_resized = tf.image.resize(image_resized, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_resized = tf.cast(image_resized, dtype=tf.uint8)
    label_string = tf.io.read_file(labelnames)
    label_resized = tf.image.decode_png(label_string)
    #label_resized = tf.image.resize(label_resized, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)/255
    mask_string = tf.io.read_file(masknames)
    mask_resized = tf.image.decode_png(mask_string)
    #mask_resized = tf.image.resize(mask_resized, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)/255
    wmask_string = tf.io.read_file(wmasknames)
    wmask_resized = tf.image.decode_png(wmask_string)
    #wmask_resized = tf.image.resize(wmask_resized, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    wmask_resized = tf.cast(wmask_resized, dtype=np.float32)
    return image_resized, label_resized/255, mask_resized/255, wmask_resized

def _decode_and_resize2(imagenames, labelnames, masknames):
    image_string = tf.io.read_file(imagenames)
    image_resized = tf.image.decode_png(image_string)
    #image_resized = tf.image.resize(image_resized, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_resized = tf.cast(image_resized, dtype=tf.uint8)
    label_string = tf.io.read_file(labelnames)
    label_resized = tf.image.decode_png(label_string)
    #label_resized = tf.image.resize(label_resized, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)/255
    mask_string = tf.io.read_file(masknames)
    mask_resized = tf.image.decode_png(mask_string)
    #mask_resized = tf.image.resize(mask_resized, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)/255
    return image_resized, label_resized/255, mask_resized/255


train_dataset = train_dataset.map(map_func=_decode_and_resize)
test_dataset = test_dataset.map(map_func=_decode_and_resize2)
#print('!')
for inum, input_one in enumerate(train_dataset):
    #print(inum)
    print(input_one[0].shape)
    print(input_one[1].shape)
    CHANNEL_NUM = input_one[0].shape[-1]
    break
train_aug_list = A.Compose([
    A.Resize(512,512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, p=0.5),
    A.Normalize(
        mean=[0.]*3,
        std=[1.]*3,
        max_pixel_value=1.0
    ),
])
val_aug_list = A.Compose([
    A.Resize(512,512),
    A.Normalize(
        mean=[0.]*3,
        std=[1.]*3,
        max_pixel_value=1.0
    ),
])


def aug_fn(x, y, m, wm):
    aug_data = train_aug_list(image=x, masks=[y, m, wm])
    x = tf.cast(aug_data['image'], dtype=tf.float32)
    y = tf.cast(aug_data['masks'][0], dtype=tf.float32)
    m = tf.cast(aug_data['masks'][1], dtype=tf.float32)
    wm = tf.cast(aug_data['masks'][2], dtype=tf.float32)
    return x, y, m, wm


def aug_fn2(x):
    aug_data = val_aug_list(image=x)
    x = tf.cast(aug_data['image'], dtype=tf.float32)
    x = tf.cast(x, dtype=tf.float32)
    return x


def process_data(x, y, m, wm):
    x, y, m, wm = tf.numpy_function(func=aug_fn, inp=[x, y, m, wm], Tout=[tf.float32, tf.float32, tf.float32, tf.float32])
    y0 = y[:, :, :1] + y[:, :, 1:2]
    y0 = tf.cast(y0 == 1, dtype=tf.float32)
    y1 = y[:, :, :1] + y[:, :, 1:2] + y[:, :, 2:]
    y1 = tf.cast(y1 > 0, dtype=tf.float32)
    y2 = y[:, :, 2:] + y[:, :, 1:2]
    y2 = tf.cast(y2 == 1, dtype=tf.float32)
    y = tf.concat([y0, y1, y2], axis=-1)
    return x, y, m, wm


def process_data2(x, y, m):
    x = tf.numpy_function(func=aug_fn2, inp=[x], Tout=tf.float32)
    y0 = y[:, :, :1] + y[:, :, 1:2]
    y0 = tf.cast(y0 == 1, dtype=tf.float32)
    y1 = y[:, :, :1] + y[:, :, 1:2] + y[:, :, 2:]
    y1 = tf.cast(y1 > 0, dtype=tf.float32)
    y2 = y[:, :, 2:] + y[:, :, 1:2]
    y2 = tf.cast(y2 == 1, dtype=tf.float32)
    y = tf.concat([y0, y1, y2], axis=-1)
    return x, y, m

train_dataset = train_dataset.map(map_func=process_data).shuffle(20)
test_dataset = test_dataset.map(map_func=process_data2)

train_dataset = train_dataset.batch(1, drop_remainder=True)
test_dataset = test_dataset.batch(1, drop_remainder=True)
model = ConnectUnet(unetn=UNETN)
x = tf.keras.Input(shape=(512,512,3), batch_size=1)
y = model(x,training = True)
#checkpoint_g = tf.train.Checkpoint(model=model)
#checkpoint_g.restore(save_path+'/model.ckpt-1')
class Crossentropy(tf.keras.losses.Loss):
    def __init__(self, mask, wm):
        self.mmm = mask
        self.mmmg = wm
        super(Crossentropy, self).__init__()

    def call(self, y_true, y_pred):
        y_true_f = (1 - y_true) * self.mmm
        yp_ = (tf.cast(y_pred <= 0.0000001, dtype=tf.int32) * tf.cast(y_true == 1, dtype=tf.int32)) == 1
        y_pred = tf.where(yp_, 0.0000001, y_pred)
        yp_2 = (tf.cast(y_pred >= 0.9999999, dtype=tf.int32) * tf.cast(y_true == 0, dtype=tf.int32)) == 1
        y_pred = tf.where(yp_2, 0.9999999, y_pred)
        mse_ = -tf.math.xlogy(y_true, y_pred) - tf.math.xlogy(y_true_f, (1 - y_pred) * self.mmm)
        mse = tf.math.reduce_sum(mse_ * self.mmmg, axis=-1)
        mse = tf.math.reduce_mean(mse)
        return mse


class Lossdice(tf.keras.losses.Loss):
    def __init__(self):
        super(Lossdice, self).__init__()

    def call(self, y_true, y_pred):
        dices = 1 - tf.reduce_mean((2 * tf.reduce_sum(y_true * y_pred, axis=[1, 2]) + 1) / (
                tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2]) + 1))
        return dices


class MixLoss(tf.keras.losses.Loss):
    def __init__(self, mask, wm):
        self.bce = Crossentropy(mask, wm)
        self.dicel = Lossdice()
        super(MixLoss, self).__init__()

    def call(self, y_true, y_pred):
        bce_loss = self.bce(y_true, y_pred)
        dicel = self.dicel(y_true, y_pred)
        return bce_loss+0.1*dicel

class MixLoss2(tf.keras.losses.Loss):
    def __init__(self, mask, wm):
        self.bce = Crossentropy(mask, wm)
        self.dicel = Lossdice()
        super(MixLoss2, self).__init__()

    def call(self, y_true, y_pred):
        bce_loss = self.bce(y_true, y_pred)
        dicel = self.dicel(y_true, y_pred)
        return bce_loss+dicel


loss = tf.keras.losses.BinaryCrossentropy()
learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000,2500,3000], [0.001, 0.0001, 0.00001, 0.000001])
optimizer0 = tf.keras.optimizers.AdamW(learning_rate=0.001)

@tf.function
def train_one_step_t(X, y, m, wm):
    with tf.GradientTape() as tape0:
        y_pred, mseg = model(X,training = True)
        loss0 = MixLoss(m, wm)(y_true=y, y_pred=y_pred[-1] * m)
        if UNETN > 1:
            loss0 += 0.1 * MixLoss2(m, wm)(y_true=y, y_pred=y_pred[0] * m)
        if UNETN > 2:
            loss0 += 0.1 * MixLoss2(m, wm)(y_true=y, y_pred=y_pred[1] * m)
    grads0 = tape0.gradient(loss0, model.trainable_variables)
    optimizer0.apply_gradients(grads_and_vars=zip(grads0, model.trainable_variables))
    return y_pred[-1], loss0


@tf.function
def test_step(x,training):
    x0in = x
    y_pred = 0
    for roti in range(3,-1,-1):
        x0in = tf.image.rot90(x0in)
        y_predrot1, mseg = model(x0in, training=True)
        y_predrot = y_predrot1[-1]
        y_pred += tf.image.rot90(y_predrot,k=roti)
    y_pred /= 4
    return y_pred, mseg

from skimage.morphology import skeletonize
if_train = 1
if if_train:
    step_num=0
    for num_epoch in range(800):
        
        losssum = 0
        metric = tf.keras.metrics.AUC(num_thresholds=2000)
        start = time.time()
        for x, y, m, wm in train_dataset:
            step_num+=1
            y_pred, lossd = train_one_step_t(x, y, m, wm)

            losssum += lossd.numpy()
            metric.update_state(y[:,:,:,::2], y_pred[:,:,:,::2])
        end = time.time()
        sys.stdout.write('epoch %d loss %.6f AUC %.4f%% -s %.2f \n' % (num_epoch, losssum / 20, metric.result().numpy() * 100, end-start))
        metric_t = tf.keras.metrics.AUC(num_thresholds=2000)
        if (num_epoch + 1) % 50 == 0 and num_epoch > 50:
            ys = []
            yps = []
            ms = []
            for i, (x, y, m) in enumerate(test_dataset):
                y_pred, _ = test_step(x, training=True)
                y_pred = tf.image.resize(y_pred, y.shape[1:3])
                y_pred = y_pred * m
                for batchi in range(y.shape[0]):
                    ys.append(y[batchi:(batchi + 1), :, :, :])
                    yps.append(y_pred[batchi:(batchi + 1), :, :, :])
                    ms.append(m[batchi:(batchi + 1), :, :, :])
            np.save('y_p2.npy', np.array(yps))
            np.save('y_t2.npy', np.array(ys))
            np.save('mask2.npy', np.array(ms))
            evaluatecal()
        if num_epoch==200:
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.save(save_path+'/model200.ckpt')
        else:
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.save(save_path+'/model.ckpt')


checkpoint_g = tf.train.Checkpoint(model=model)
checkpoint_g.restore(save_path+'/model.ckpt-1')

metric_t = tf.keras.metrics.AUC(num_thresholds=2000)
for x, y, m in test_dataset:

    y_pred, mseg = test_step(x,training = True)
    y_pred = tf.image.resize(y_pred, y.shape[1:3])
    y_pred = y_pred * m
    metric_t.update_state(y[:,:,:,::2], y_pred[:,:,:,::2])
best_result=1*metric_t.result().numpy()
print('best_result', best_result)

ys = []
yps = []
ms = []
losssum = 0
u = 0
d = 0
import shutil

for i, (x, y, m) in enumerate(test_dataset):
    y_pred, _ = test_step(x,training = True)
    y_pred = tf.image.resize(y_pred, y.shape[1:3])
    print(y_pred.shape)
    y_pred = y_pred*m
    losssum+=loss(y, y_pred).numpy()
    uu = tf.cast(y_pred[:,:,:,::2]>0.5, dtype=tf.float32)+y[:,:,:,::2]
    u += tf.reduce_sum(tf.cast(uu==0, dtype=tf.float32))+tf.reduce_sum(tf.cast(uu==2, dtype=tf.float32))
    d += y_pred.shape[0]*y_pred.shape[1]*y_pred.shape[2]*2
    cv2.imwrite('RITE/result/%d.png'%i,np.array((y_pred[0,:,:,:]).numpy()*255, dtype=np.uint8))
    cv2.imwrite('RITE/result_true/%d.png'%i,np.array((y[0,:,:,:]).numpy()*255, dtype=np.uint8))
    for batchi in range(y.shape[0]):
        ys.append(y[batchi:(batchi+1),:,:,:])
        yps.append(y_pred[batchi:(batchi+1),:,:,:])
        ms.append(m[batchi:(batchi+1),:,:,:])
np.save('y_p2.npy',np.array(yps))
np.save('y_t2.npy',np.array(ys))
np.save('mask2.npy',np.array(ms))
evaluatecal()
sys.stdout.write('val_loss %.6f\n' % (losssum/10))
#sys.stdout.close()
#sys.stdout = sys.__stdout__
