import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import time
from connectunets import ConnectUnet
#from nunet import ConnectUnet
from sklearn.metrics import roc_auc_score
import logging
from torchinfo import summary
import re
import h5py
import results_config
from scipy.ndimage import zoom
from result_indcal import evaluatecal

def get_results(flod):
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print(torch.cuda.get_device_name(0))
    else:
        print("cpu")

    if not os.path.exists('./result'):
        os.makedirs('./result')
    path = results_config.data_path
    if not os.path.exists('./num_rand.npy'):
        np.random.seed(10)
        num_ = np.arange(2594)
        np.random.shuffle(num_)
        np.save('./num_rand.npy', num_)
    else:
        num_ = np.load('./num_rand.npy')
    if (not os.path.exists('./trainimages.npy')) or (not os.path.exists('./testimages.npy')):
        trainimagesa = os.listdir(path + '/ISIC2018_Task1-2_Training_Input_png')
        trainimagesa = list(filter(lambda x: x[-3:] == 'png', trainimagesa))
        trainimagesa = np.array(list(trainimagesa))[num_]
        trainimages = trainimagesa[:2075]
        np.save('./trainimages.npy', trainimages)
        testimages = trainimagesa[2075:]
        np.save('./testimages.npy', testimages)
    else:
        trainimages = np.load('./trainimages.npy')
        testimages = np.load('./testimages.npy')

    # testimages = os.listdir(path + '/ISIC2018_Task1-2_Test_Input')
    # testimages = list(filter(lambda x: x[-3:]=='jpg', testimages))
    train_imagenames = [path + '/ISIC2018_Task1-2_Training_Input_png/' + name for name in trainimages]
    test_imagenames = [path + '/ISIC2018_Task1-2_Training_Input_png/' + name for name in testimages]
    train_labelnames = [path + '/ISIC2018_Task1_Training_GroundTruth_png/' + name[:-4] + '_segmentation.png' for name in
                        trainimages]
    test_labelnames = [path + '/ISIC2018_Task1_Training_GroundTruth_png/' + name[:-4] + '_segmentation.png' for name in
                       testimages]

    transform_v = A.Compose([
        # A.Resize(256,256),
        A.Normalize(
            mean=[0.] * 3,
            std=[1.] * 3,
            max_pixel_value=255.0
        ),
        ToTensorV2(transpose_mask=True, p=1.0)
    ])

    class ISICDataset(Dataset):
        def __init__(self, imagenames=train_imagenames, labelnames=train_labelnames, transform=transform_v):
            self.imagenames = imagenames
            self.labelnames = labelnames
            self.transform = transform

        def __len__(self):
            return len(self.imagenames)

        def __getitem__(self, idx):
            while True:
                image_resized = cv2.imread(self.imagenames[idx])
                if not image_resized is None:

                    break
                else:
                    time.sleep(1)
            label_resized = cv2.imread(self.labelnames[idx], 0)
            label_resized = np.expand_dims(label_resized, -1)

            if image_resized is None:
                print(self.imagenames[idx])
            aug_data = self.transform(image=image_resized, mask=label_resized)
            x = aug_data['image'].float()
            y = aug_data['mask'].float() / 255.
            nameout = self.imagenames[idx].split('/')[-1][:-4]
            # y = y.unsqueeze(0)
            return x, y, nameout

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    unetn = results_config.nunet
    model = ConnectUnet(unetn, results_config.base_filters)
    #summary_model = summary(model, input_size=(1, 3, 256, 256))
    # print(summary(model, input_size=(1, 3, 256,256)))
    #logging.info(summary_model)
    # model_pra = reversed(model.state_dict().items())
    # next(model_pra)
    # print(next(model_pra)[-1].detach().to(device='cpu').numpy()[:,:,0,0].T)
    model.load_state_dict(torch.load('./%d/pysave/model_d%d_c%d.pth' % (flod, unetn, results_config.base_filters), weights_only=True))
    model = model.to(device=device)

    test_ds = ISICDataset(test_imagenames, test_labelnames, transform=transform_v)
    test_dl = DataLoader(
        test_ds,
        batch_size=results_config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    featuremapsavepath = './isic_result/featuremap_d%d_c%d' % (unetn, results_config.base_filters)
    if not os.path.exists(featuremapsavepath):
        os.makedirs(featuremapsavepath)

    predsavepath = './isic_result/pred_image_d%d_c%d' % (unetn, results_config.base_filters)
    if not os.path.exists(predsavepath):
        os.makedirs(predsavepath)
    if not os.path.exists('./fmname'):
        os.makedirs('./fmname')
    # ============ 新增：按 tag + fold 分开存盘 ============
    tag = 'd%d_c%d' % (unetn, results_config.base_filters)
    result_dir = './result/%s/fold%d' % (tag, flod)
    os.makedirs(result_dir, exist_ok=True)
    # ====================================================

    with torch.no_grad():
        for nums, (x, y, nameout) in enumerate(tqdm(test_dl)):
            np.save('./fmname/%d.npy' % nums, nameout)
            y_pred = 0
            for roti in range(1, 5, 1):
                x = x.to(device=device)
                x0in = torch.rot90(x, k=roti, dims=(2, 3))
                y_predrot1, mseg = model(x0in)
                np.save(featuremapsavepath + '/%d.npy' % nums,
                        mseg.detach().to(device='cpu').numpy())
                y_predrot = y_predrot1[-1]
                y_predrot = torch.rot90(y_predrot, k=4 - roti, dims=(2, 3)) \
                                 .detach().to(device='cpu')
                for batchi in range(y.shape[0]):
                    np.save(predsavepath + '/%d.npy' % (y.shape[0] * nums + batchi),
                            y_predrot[batchi, :, :, :].numpy().transpose(1, 2, 0))
                y_pred += y_predrot
            y_pred /= 4
            y_pred = F.interpolate(y_pred, size=[y.shape[-2], y.shape[-1]],
                                   mode='bilinear').numpy()
            y = y.numpy()
            for batchi in range(y.shape[0]):
                y_preds = np.reshape(y_pred[batchi, 0, :, :], [-1]).astype(np.float32)
                y_trues = np.reshape(y[batchi, 0, :, :], [-1]).astype(np.float32)
                # === 关键改动：保存 [概率, 硬标签]，而非 [>0.5, >0.5] ===
                # 下游 evaluatecal 同时用概率（算 AUC）和阈值（算 MCM）
                np.save(os.path.join(result_dir, nameout[batchi] + '.npy'),
                        [y_preds, y_trues])
                # 兼容原代码：如果想保留原路径的硬标签副本，取消注释
                # yc = [y_preds > 0.5, y_trues > 0.5]
                # np.save('./result/' + nameout[batchi] + '.npy', yc)

    # ============ 调用 evaluatecal 时传入新目录 ============
    ind = evaluatecal(filename='_d%d_c%d_%d' % (
                          unetn, results_config.base_filters, flod),
                      result_dir=result_dir,
                      per_image_dir='./isic_result/per_image/%s/fold%d' % (tag, flod))
    return ind


if __name__ == '__main__':
    import csv
    if os.path.exists('ind1.csv'):
        os.remove('ind1.csv')
    flods = [1, 2, 3, 4, 5]
    metrall = []
    for flod in flods:
        metrall.append(1*get_results(flod))
    metravg = np.average(metrall,axis=0)
    metrstd = np.std(metrall,axis=0)
    with open('ind1.csv', mode='a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['d%d_c%d'%(results_config.nunet, results_config.base_filters)])
        csv_writer.writerow(metravg)
        csv_writer.writerow(metrstd)

    results_config.nunet=3
    results_config.base_filters=32
    metrall = []
    for flod in flods:
        metrall.append(1*get_results(flod))
    metravg = np.average(metrall,axis=0)
    metrstd = np.std(metrall,axis=0)
    with open('ind1.csv', mode='a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['d%d_c%d'%(results_config.nunet, results_config.base_filters)])
        csv_writer.writerow(metravg)
        csv_writer.writerow(metrstd)

    results_config.nunet=2
    results_config.base_filters=64
    metrall = []
    for flod in flods:
        metrall.append(1*get_results(flod))
    metravg = np.average(metrall,axis=0)
    metrstd = np.std(metrall,axis=0)
    with open('ind1.csv', mode='a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['d%d_c%d'%(results_config.nunet, results_config.base_filters)])
        csv_writer.writerow(metravg)
        csv_writer.writerow(metrstd)

    results_config.nunet=2
    results_config.base_filters=32
    metrall = []
    for flod in flods:
        metrall.append(1*get_results(flod))
    metravg = np.average(metrall,axis=0)
    metrstd = np.std(metrall,axis=0)
    with open('ind1.csv', mode='a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['d%d_c%d'%(results_config.nunet, results_config.base_filters)])
        csv_writer.writerow(metravg)
        csv_writer.writerow(metrstd)

    results_config.nunet=1
    results_config.base_filters=64
    metrall = []
    for flod in flods:
        metrall.append(1*get_results(flod))
    metravg = np.average(metrall,axis=0)
    metrstd = np.std(metrall,axis=0)
    with open('ind1.csv', mode='a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['d%d_c%d'%(results_config.nunet, results_config.base_filters)])
        csv_writer.writerow(metravg)
        csv_writer.writerow(metrstd)

    results_config.nunet=1
    results_config.base_filters=32
    metrall = []
    for flod in flods:
        metrall.append(1*get_results(flod))
    metravg = np.average(metrall,axis=0)
    metrstd = np.std(metrall,axis=0)
    with open('ind1.csv', mode='a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['d%d_c%d'%(results_config.nunet, results_config.base_filters)])
        csv_writer.writerow(metravg)
        csv_writer.writerow(metrstd)
    