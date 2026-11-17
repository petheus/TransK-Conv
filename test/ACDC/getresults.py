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
    if not os.path.exists('./trueresult'):
        os.makedirs('./trueresult')

    # path = '/home/ultramank/paper3/acdcd/data/ACDC'
    path = results_config.data_path
    trainimages = ["patient{:0>3}".format(i) for i in
                   [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                    71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90]]
    testimages = ["patient{:0>3}".format(i) for i in [5, 39, 77, 82, 78, 10, 64, 24, 30, 73, 80, 41, 36, 60, 72]]

    train_slices = os.listdir(
        path + "/ACDC_training_slices")
    train_imagenames = []
    for ids in trainimages:
        new_data_list = list(filter(lambda x: re.match(
            '{}.*'.format(ids), x) != None, train_slices))
        train_imagenames.extend(new_data_list)

    test_volumes = os.listdir(
        path + "/ACDC_training_volumes")
    test_imagenames = []
    for ids in testimages:
        new_data_list = list(filter(lambda x: re.match(
            '{}.*'.format(ids), x) != None, test_volumes))
        test_imagenames.extend(new_data_list)

    transform_v = A.Compose([
        # A.Resize(256,256),
        ToTensorV2(transpose_mask=True, p=1.0)
    ])

    class ACDCDataset(Dataset):
        def __init__(self, imagenames=train_imagenames, split="train", transform=None):
            self.imagenames = imagenames
            self.split = split
            self.transform = transform
            self.output_size = [256, 256]

        def __len__(self):
            return len(self.imagenames)

        def __getitem__(self, idx):
            case = self.imagenames[idx]
            if self.split == "train":
                h5f = h5py.File(path +
                                "/ACDC_training_slices/{}".format(case), 'r')
            else:
                h5f = h5py.File(path +
                                "/ACDC_training_volumes/{}".format(case), 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}
            if self.split == "train":
                image = h5f['image'][:]
                label = h5f["label"][:]
                x, y = image.shape
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
                image = np.expand_dims(image, -1)
                label = np.expand_dims(label, -1)

                aug_data = self.transform(image=image, mask=label)
                image = aug_data['image'].float()
                label = torch.permute(F.one_hot(aug_data['mask'].long()[0, :, :], 4).float(), (2, 0, 1))
                return image, label, case

            else:
                image = h5f['image'][:]

                label = h5f['label'][:]
                _, x, y = image.shape
                image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
                # label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
                image = np.expand_dims(image, 1)
                # label = np.expand_dims(label,1)
                # label = torch.permute(F.one_hot(label, 4), (0,3,1,2))
                image = torch.from_numpy(image).float()
                label = torch.from_numpy(label).float()
                return image, label, case

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    unetn = results_config.nunet
    model = ConnectUnet(unetn, results_config.base_filters)

    model.load_state_dict(torch.load('./%d/pysave/model_d%d_c%d.pth' % (flod, unetn, results_config.base_filters), weights_only=True))
    model = model.to(device=device)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 10, T_mult=2, eta_min=0.0002)
    test_ds = ACDCDataset(test_imagenames, "test", transform=transform_v)

    featuremapsavepath = './acdc_result/featuremap_d%d_c%d' % (unetn, results_config.base_filters)
    if not os.path.exists(featuremapsavepath):
        os.makedirs(featuremapsavepath)

    predsavepath = './acdc_result/pred_image_d%d_c%d' % (unetn, results_config.base_filters)
    if not os.path.exists(predsavepath):
        os.makedirs(predsavepath)

    tag = 'd%d_c%d' % (unetn, results_config.base_filters)
    result_dir     = './result/%s/fold%d'     % (tag, flod)
    trueresult_dir = './trueresult/%s/fold%d' % (tag, flod)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(trueresult_dir, exist_ok=True)
    # ======================================================

    with torch.no_grad():
        for nums in range(len(test_imagenames)):
            x, y, nameout = test_ds[nums]
            print(nameout)
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
                y_pred += y_predrot
            y_pred /= 4
            y_pred = F.interpolate(y_pred, size=[y.shape[-2], y.shape[-1]],
                                   mode='bilinear').numpy()
            y = y.numpy()
            y_trues = np.reshape(y, [-1])

            # ============ ② 保存路径改为新目录 ============
            np.save(os.path.join(trueresult_dir, nameout + '.npy'),
                    y_trues.astype(np.uint8))
            np.save(os.path.join(result_dir, nameout + '.npy'), y_pred)
            # ============================================

    # ============ ③ 调用 evaluatecal 传入新目录 ============
    ind = evaluatecal(
        filename='_d%d_c%d_%d' % (unetn, results_config.base_filters, flod),
        result_dir=result_dir,
        trueresult_dir=trueresult_dir,
        per_image_dir='./acdc_result/per_image/%s/fold%d' % (tag, flod),
    )
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
    