import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from net.twgan import GeneratorNet, VGGNet, DescriminatorNet
import albumentations as A
import numpy as np
import cv2
import sys
import time
from tqdm import tqdm
from indCal import evaluatecal
#sys.stdout = open('output.txt', 'w')
model_save_path = './twgansave'
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
path = './RITE'
np.random.seed(10)
num_ = np.arange(20)
np.random.shuffle(num_)
num_ = num_
trainimages = os.listdir(path + '/training/images_png')
trainimages = np.array(list(trainimages))
trainimages = trainimages[num_]
testimages = os.listdir(path + '/test/images_png')
train_imagenames = tf.constant([path + '/training/images_png/' + name for name in trainimages])
test_imagenames = tf.constant([path + '/test/images_png/' + name for name in testimages])
train_labelnames = tf.constant([path + '/training/av/' + name for name in trainimages])
train_labeldil5names = tf.constant([path + '/dil/5_' + name for name in trainimages])
train_labeldil9names = tf.constant([path + '/dil/9_' + name for name in trainimages])
train_shufflednames = tf.constant([path + '/shuffled/' + name for name in trainimages])

test_labelnames = tf.constant([path + '/test/av/' + name for name in testimages])
train_masknames = tf.constant([path + '/training/mask_png/' + name for name in trainimages])
test_masknames = tf.constant([path + '/test/mask_png/' + name for name in testimages])
train_wmasknames = tf.constant([path + '/training/weight_label/' + name for name in trainimages])
train_dataset = tf.data.Dataset.from_tensor_slices((train_imagenames, train_labelnames, train_masknames, train_wmasknames, train_labeldil5names,train_labeldil9names, train_shufflednames))
test_dataset = tf.data.Dataset.from_tensor_slices((test_imagenames, test_labelnames, test_masknames))


def _decode_and_resize(imagenames, labelnames, masknames, wmasknames,labeldil5names,labeldil9names,labelshufflednames):
    image_string = tf.io.read_file(imagenames)
    image_resized = tf.image.decode_png(image_string)
    #image_resized = tf.image.resize(image_resized, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_resized = tf.cast(image_resized[36:548,26:538,:], dtype=tf.uint8)
    label_string = tf.io.read_file(labelnames)
    label_resized = tf.image.decode_png(label_string)[36:548,26:538,:]
    #label_resized = tf.image.resize(label_resized, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)/255
    mask_string = tf.io.read_file(masknames)
    mask_resized = tf.image.decode_png(mask_string)[36:548,26:538,:]
    #mask_resized = tf.image.resize(mask_resized, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)/255
    wmask_string = tf.io.read_file(wmasknames)
    wmask_resized = tf.image.decode_png(wmask_string)[36:548,26:538,:]
    #wmask_resized = tf.image.resize(wmask_resized, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    wmask_resized = tf.cast(wmask_resized, dtype=np.float32)
    label5dil_string = tf.io.read_file(labeldil5names)
    label5dil_resized = tf.image.decode_png(label5dil_string)[36:548,26:538,:]
    label9dil_string = tf.io.read_file(labeldil9names)
    label9dil_resized = tf.image.decode_png(label9dil_string)[36:548,26:538,:]

    labelshuffled_string = tf.io.read_file(labelshufflednames)
    labelshuffled_resized = tf.image.decode_png(labelshuffled_string)[36:548,26:538,:]
    return image_resized, label_resized/255, mask_resized/255, wmask_resized, label5dil_resized/255, label9dil_resized/255, labelshuffled_resized/255

def _decode_and_resize2(imagenames, labelnames, masknames):
    image_string = tf.io.read_file(imagenames)
    image_resized = tf.image.decode_png(image_string)[36:548,26:538,:]
    #image_resized = tf.image.resize(image_resized, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_resized = tf.cast(image_resized, dtype=tf.uint8)
    label_string = tf.io.read_file(labelnames)
    label_resized = tf.image.decode_png(label_string)[36:548,26:538,:]
    #label_resized = tf.image.resize(label_resized, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)/255
    mask_string = tf.io.read_file(masknames)
    mask_resized = tf.image.decode_png(mask_string)[36:548,26:538,:]
    #mask_resized = tf.image.resize(mask_resized, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)/255
    return image_resized, label_resized/255, mask_resized/255


train_dataset = train_dataset.map(map_func=_decode_and_resize)
test_dataset = test_dataset.map(map_func=_decode_and_resize2)
#print('!')
for inum, input_one in enumerate(train_dataset):
    #print(inum)
    print(input_one[0].shape)
    print(input_one[1].shape)
    
    INPUT_IMAGE_SIZE = [input_one[0].shape[0],input_one[0].shape[1]]
    CHANNEL_NUM = input_one[0].shape[-1]
    break
train_aug_list = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, p=0.5),

    A.Normalize(
        mean=[0.]*3,
        std=[1.]*3,
        max_pixel_value=1.0
    ),
])
val_aug_list = A.Compose([
    A.Normalize(
        mean=[0.]*3,
        std=[1.]*3,
        max_pixel_value=1.0
    ),
])


def aug_fn(x, y, m, wm, y5dil,y9dil,yshuffled):
    aug_data = train_aug_list(image=x, masks=[y, m, wm, y5dil,y9dil,yshuffled])
    x = tf.cast(aug_data['image'], dtype=tf.float32)
    y = tf.cast(aug_data['masks'][0], dtype=tf.float32)
    m = tf.cast(aug_data['masks'][1], dtype=tf.float32)
    wm = tf.cast(aug_data['masks'][2], dtype=tf.float32)
    y5dil = tf.cast(aug_data['masks'][3], dtype=tf.float32)
    y9dil = tf.cast(aug_data['masks'][4], dtype=tf.float32)
    yshuffled = tf.cast(aug_data['masks'][5], dtype=tf.float32)
    return x, y, m, wm, y5dil,y9dil,yshuffled


def aug_fn2(x):
    aug_data = val_aug_list(image=x)
    x = tf.cast(aug_data['image'], dtype=tf.float32)
    x = tf.cast(x, dtype=tf.float32)
    return x


def process_data(x, y, m, wm, y5dil,y9dil,yshuffled):
    x, y, m, wm, y5dil,y9dil,yshuffled = tf.numpy_function(func=aug_fn, inp=[x, y, m, wm, y5dil,y9dil,yshuffled], Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
    y0 = y[:, :, :1] + y[:, :, 1:2]
    y0 = tf.cast(y0 == 1, dtype=tf.float32)
    y1 = y[:, :, :1] + y[:, :, 1:2] + y[:, :, 2:]
    y1 = tf.cast(y1 > 0, dtype=tf.float32)
    y2 = y[:, :, 2:] + y[:, :, 1:2]
    y2 = tf.cast(y2 == 1, dtype=tf.float32)
    y = tf.concat([y0, y1, y2], axis=-1)
    return x, y, m, wm, y5dil,y9dil,yshuffled


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

batchsize=1
train_dataset = train_dataset.batch(batchsize, drop_remainder=False)
test_dataset = test_dataset.batch(batchsize, drop_remainder=False)
modelG = GeneratorNet()
modelD = DescriminatorNet()
modelVGG = VGGNet()
x = tf.keras.Input(shape=(512,512,3), batch_size=1)
y = modelG(x,training = True)
x2 = tf.keras.Input(shape=(512,512,6), batch_size=1)
y1 = modelD(x2,training = True)
y2 = modelVGG(x,training = True)
#checkpoint_g = tf.train.Checkpoint(model=modelG)
#checkpoint_g.restore(model_save_path+'/modelG.ckpt-1')
class Crossentropy(tf.keras.losses.Loss):
    def __init__(self, reduction='none'):
        super(Crossentropy, self).__init__(reduction=reduction)

    def call(self, y_true, y_pred):
        y_true_f = (1 - y_true)
        yp_ = (tf.cast(y_pred <= 0.0000001, dtype=tf.int32) * tf.cast(y_true == 1, dtype=tf.int32)) == 1
        y_pred = tf.where(yp_, 0.0000001, y_pred)
        yp_2 = (tf.cast(y_pred >= 0.9999999, dtype=tf.int32) * tf.cast(y_true == 0, dtype=tf.int32)) == 1
        y_pred = tf.where(yp_2, 0.9999999, y_pred)
        mse = -tf.math.xlogy(y_true, y_pred) - tf.math.xlogy(y_true_f, (1 - y_pred))
        return mse
BCEloss = tf.keras.losses.BinaryCrossentropy(reduction='none')
BCElossno = Crossentropy(reduction='none')
huberloss = tf.keras.losses.Huber(reduction='none')
#metric = tf.keras.metrics.AUC(num_thresholds=2000)
#modelsch = tf.keras.optimizers.schedules.CosineDecayRestarts(0.002, 2001, alpha=0.1,t_mul=2.0,m_mul=0.5)
lrinit=0.001
lrinitg=0.001
epoch_init = 7000
learning_rate_fng = tf.keras.optimizers.schedules.PiecewiseConstantDecay([epoch_init,epoch_init*2,epoch_init*4], [lrinitg, lrinitg/2, lrinitg/4, lrinitg/8])
learning_rate_fnd = tf.keras.optimizers.schedules.PiecewiseConstantDecay([epoch_init,epoch_init*2,epoch_init*4], [lrinit, lrinit/2, lrinit/4, lrinit/8])
learning_rate_fndd = tf.keras.optimizers.schedules.PiecewiseConstantDecay([epoch_init,epoch_init*2,epoch_init*4], [lrinit, lrinit/2, lrinit/4, lrinit/8])
optimizerg = tf.keras.optimizers.AdamW(learning_rate=lrinitg)
optimizervgg = tf.keras.optimizers.AdamW(learning_rate=lrinit)
optimizerd = tf.keras.optimizers.AdamW(learning_rate=lrinit)

classweight = tf.constant([[0.3,0.4,0.3]])

@tf.function
def train_one_step_t(X, y, m, wm, y5dil,y9dil,yshuffled):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        y_pred, mseg = modelG(X,training = True)
        gout = modelD(tf.concat([X,y_pred],axis=-1),training = True)
        goutm = modelD(tf.concat([X,y],axis=-1),training = True)
        goutms = modelD(tf.concat([X,yshuffled],axis=-1),training = True)
        vggout0 = modelVGG(tf.repeat(y_pred[:,:,:,:1], [3], axis=-1),training = True)
        vggout1 = modelVGG(tf.repeat(y_pred[:,:,:,1:2], [3], axis=-1),training = True)
        vggout2 = modelVGG(tf.repeat(y_pred[:,:,:,2:3], [3], axis=-1),training = True)
        vggoutm0 = modelVGG(tf.repeat(y[:,:,:,:1], [3], axis=-1),training = True)
        vggoutm1 = modelVGG(tf.repeat(y[:,:,:,1:2], [3], axis=-1),training = True)
        vggoutm2 = modelVGG(tf.repeat(y[:,:,:,2:3], [3], axis=-1),training = True)
        vggoutms0 = modelVGG(tf.repeat(yshuffled[:,:,:,:1], [3], axis=-1),training = True)
        vggoutms1 = modelVGG(tf.repeat(yshuffled[:,:,:,1:2], [3], axis=-1),training = True)
        vggoutms2 = modelVGG(tf.repeat(yshuffled[:,:,:,2:3], [3], axis=-1),training = True)
        class_sizes = gout.shape[:3]
        class1 = tf.zeros([*class_sizes, 2], dtype=tf.float32)
        class2 = tf.concat([tf.ones([*class_sizes, 1], dtype=tf.float32),tf.zeros([*class_sizes, 1], dtype=tf.float32)],axis=-1)
        class3 = tf.ones([*class_sizes, 2], dtype=tf.float32)
        ladvd = tf.math.reduce_mean(tf.math.reduce_mean(BCElossno(y_true=class1, y_pred=goutms)+BCElossno(y_true=class2, y_pred=gout)+BCElossno(y_true=class3, y_pred=goutm), axis=0))
        ltri = 0
        for vggi in range(4):
            tonm = vggoutm0[vggi].shape[1]*vggoutm0[vggi].shape[2]*vggoutm0[vggi].shape[3]
            ldi0 = tf.math.reduce_mean(tf.math.maximum(tf.math.sqrt(tf.math.maximum(tf.math.reduce_sum(tf.math.square(vggoutm0[vggi]-vggout0[vggi]),axis=[1,2,3]),1e-10))/tonm-tf.math.sqrt(tf.math.maximum(tf.math.reduce_sum(tf.math.square(vggoutm0[vggi]-vggoutms0[vggi]),axis=[1,2,3]),1e-10))/tonm+1., 0.))
            ldi1 = tf.math.reduce_mean(tf.math.maximum(tf.math.sqrt(tf.math.maximum(tf.math.reduce_sum(tf.math.square(vggoutm1[vggi]-vggout1[vggi]),axis=[1,2,3]),1e-10))/tonm-tf.math.sqrt(tf.math.maximum(tf.math.reduce_sum(tf.math.square(vggoutm1[vggi]-vggoutms1[vggi]),axis=[1,2,3]),1e-10))/tonm+1., 0.))
            ldi2 = tf.math.reduce_mean(tf.math.maximum(tf.math.sqrt(tf.math.maximum(tf.math.reduce_sum(tf.math.square(vggoutm2[vggi]-vggout2[vggi]),axis=[1,2,3]),1e-10))/tonm-tf.math.sqrt(tf.math.maximum(tf.math.reduce_sum(tf.math.square(vggoutm2[vggi]-vggoutms2[vggi]),axis=[1,2,3]),1e-10))/tonm+1., 0.))
            ltri += (ldi0+ldi1+ldi2)/3
        ltri /= 4
        lwidth = tf.math.reduce_mean(tf.math.reduce_sum(huberloss(y_true=y9dil, y_pred=mseg[:,:,:,:3]*y9dil), axis=[1,2])/tf.math.reduce_sum(y9dil, axis=[1,2,3])*3
                                    +tf.math.reduce_sum(huberloss(y_true=y5dil, y_pred=mseg[:,:,:,3:6]*y5dil), axis=[1,2])/tf.math.reduce_sum(y5dil, axis=[1,2,3])*3
                                    +tf.math.reduce_sum(huberloss(y_true=y, y_pred=mseg[:,:,:,6:]*y), axis=[1,2])/tf.math.reduce_sum(y, axis=[1,2,3])*3)
        lbce = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.reduce_mean(BCElossno(y_true=y, y_pred=y_pred), axis=[1,2])*classweight,axis=-1))
        ladvg = tf.math.reduce_mean(tf.math.reduce_mean(BCElossno(y_true=class3, y_pred=gout), axis=0))
        lossgen = lbce+0.01*ladvg+0.2*lwidth+0.2*ltri
        lossdis = 0.01*ladvd
    gradsG = gen_tape.gradient(lossgen, modelG.trainable_variables)
    optimizervgg.apply_gradients(grads_and_vars=zip(gradsG, modelG.trainable_variables))
    gradsD = disc_tape.gradient(lossdis, modelD.trainable_variables)
    optimizerd.apply_gradients(grads_and_vars=zip(gradsD, modelD.trainable_variables))
    return y_pred, lossgen

@tf.function
def train_one_step_tw(X, y, m, wm, y5dil,y9dil,yshuffled):
    with tf.GradientTape() as gen_tape:
        y_pred, mseg = modelG(X,training = True)
        lwidth = tf.math.reduce_mean(tf.math.reduce_sum(huberloss(y_true=y9dil, y_pred=mseg[:,:,:,:3]*y9dil), axis=[1,2])/tf.math.reduce_mean(tf.math.reduce_sum(y9dil, axis=[1,2]), axis=-1)
                                    +tf.math.reduce_sum(huberloss(y_true=y5dil, y_pred=mseg[:,:,:,3:6]*y5dil), axis=[1,2])/tf.math.reduce_mean(tf.math.reduce_sum(y5dil, axis=[1,2]), axis=-1)
                                    +tf.math.reduce_sum(huberloss(y_true=y, y_pred=mseg[:,:,:,6:]*y), axis=[1,2])/tf.math.reduce_mean(tf.math.reduce_sum(y, axis=[1,2]), axis=-1))
        lbce = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.reduce_mean(BCElossno(y_true=y, y_pred=y_pred), axis=[1,2])*classweight, axis=-1))
        lossgen = lbce+0.2*lwidth
    gradsG = gen_tape.gradient(lossgen, modelG.trainable_variables)
    optimizerg.apply_gradients(grads_and_vars=zip(gradsG, modelG.trainable_variables))
    return y_pred, lossgen


@tf.function
def test_step(x,training):
    x0in = x
    y_pred = 0
    for roti in range(3,-1,-1):
        x0in = tf.image.rot90(x0in)
        y_predrot1, mseg = modelG(x0in, training=True)
        y_predrot = y_predrot1
        y_pred += tf.image.rot90(y_predrot,k=roti)
    y_pred /= 4
    return y_pred, mseg

from skimage.morphology import skeletonize
if_train = 1
if if_train:
    step_num=0
    for num_epoch in range(500):
        start = time.time()
        losssum = 0
        metric = tf.keras.metrics.AUC(num_thresholds=2000)
        for x, y, m, wm, y5dil,y9dil,yshuffled in train_dataset:
            y_pred, lossd = train_one_step_t(x, y, m, wm, y5dil,y9dil,yshuffled)

            losssum += lossd.numpy()
            metric.update_state(y[:,:,:,::2], y_pred[:,:,:,::2])

        end = time.time()
        sys.stdout.write('epoch %d loss %.6f AUC %.4f%% -s %.2f ' % (num_epoch, losssum / 20, metric.result().numpy() * 100, end-start))
        metric_t = tf.keras.metrics.AUC(num_thresholds=2000)
        for x, y, m in test_dataset:

            y_pred, mseg = test_step(x,training = True)
            y_pred = y_pred * m
            losssum += tf.math.reduce_mean(BCElossno(y_true=y, y_pred=y_pred)).numpy()
            metric_t.update_state(y[:,:,:,::2], y_pred[:,:,:,::2])
        checkpoint = tf.train.Checkpoint(model=modelG)
        checkpoint.save(model_save_path+'/modelG.ckpt')
        checkpoint = tf.train.Checkpoint(model=modelD)
        checkpoint.save(model_save_path+'/modelD.ckpt')
        checkpoint = tf.train.Checkpoint(model=modelVGG)
        checkpoint.save(model_save_path+'/modelVGG.ckpt')
        if (num_epoch+1)%50==0 and num_epoch>=0:
            ys = []
            yps = []
            ms = []
            for i, (x, y, m) in enumerate(test_dataset):
                y_pred, _ = test_step(x,training = True)
                y_pred = y_pred*m
                for batchi in range(y.shape[0]):
                    ys.append(y[batchi:(batchi+1),:,:,:])
                    yps.append(y_pred[batchi:(batchi+1),:,:,:])
                    ms.append(m[batchi:(batchi+1),:,:,:])
            np.save('y_p2.npy',np.array(yps))
            np.save('y_t2.npy',np.array(ys))
            np.save('mask2.npy',np.array(ms))
            evaluatecal()
        if num_epoch==200:
            checkpoint = tf.train.Checkpoint(model=modelG)
            checkpoint.save(model_save_path+'/modelG200.ckpt')
            checkpoint = tf.train.Checkpoint(model=modelD)
            checkpoint.save(model_save_path+'/modelD200.ckpt')
            checkpoint = tf.train.Checkpoint(model=modelVGG)
            checkpoint.save(model_save_path+'/modelVGG200.ckpt')
        if 'best_result' in dir():
            if best_result<metric_t.result().numpy():
                checkpoint = tf.train.Checkpoint(model=modelG)
                checkpoint.save(model_save_path+'/modelG_best2.ckpt')
                checkpoint = tf.train.Checkpoint(model=modelD)
                checkpoint.save(model_save_path+'/modelD_best2.ckpt')
                checkpoint = tf.train.Checkpoint(model=modelVGG)
                checkpoint.save(model_save_path+'/modelVGG_best2.ckpt')
                best_result=1*metric_t.result().numpy()
                end = time.time()
                sys.stdout.write('val_loss %.6f val_AUC %.4f%% -s %.2f -save best\n' % (losssum / 20, metric_t.result().numpy() * 100, end-start))
            else:
                end = time.time()
                sys.stdout.write('val_loss %.6f val_AUC %.4f%% -s %.2f\n' % (losssum / 20, metric_t.result().numpy() * 100, end-start))
        else:
            checkpoint = tf.train.Checkpoint(model=modelG)
            checkpoint.save(model_save_path+'/modelG.ckpt')
            checkpoint = tf.train.Checkpoint(model=modelD)
            checkpoint.save(model_save_path+'/modelD.ckpt')
            checkpoint = tf.train.Checkpoint(model=modelVGG)
            checkpoint.save(model_save_path+'/modelVGG.ckpt')
            best_result=1*metric_t.result().numpy()
            end = time.time()
            sys.stdout.write('val_loss %.6f val_AUC %.4f%% -s %.2f -save best\n' % (losssum / 20, metric_t.result().numpy() * 100, end-start))
    print('best_result', best_result)

checkpoint_g = tf.train.Checkpoint(model=modelG)
checkpoint_g.restore(model_save_path+'/modelG.ckpt-1')

metric_t = tf.keras.metrics.AUC(num_thresholds=2000)
for x, y, m in test_dataset:

    y_pred, mseg = test_step(x,training = True)
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
    print(y_pred.shape)
    y_pred = y_pred*m
    losssum+=tf.math.reduce_mean(BCElossno(y, y_pred)).numpy()
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
