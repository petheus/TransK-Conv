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
from scipy.ndimage.interpolation import zoom
from result_indcal import evaluatecal

def get_results(flod):
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print(torch.cuda.get_device_name(0))
    else:
        print("cpu")

    #flod = results_config.flod
    path = results_config.data_path  # './RITE'
    np.random.seed(10)
    testimages = os.listdir(path + '/test/images_png')
    testimages = list(filter(lambda x: x[-3:] == 'png', testimages))
    testimages.sort()
    print(testimages)
    test_imagenames = [path + '/test/images_png/' + name for name in testimages]

    test_labelnames = [path + '/test/av/' + name for name in testimages]
    test_masknames = [path + '/test/mask_png/' + name for name in testimages]

    transform_v = A.Compose([
        A.Normalize(
            mean=[0.] * 3,
            std=[1.] * 3,
            max_pixel_value=1.0
        ),
        ToTensorV2(transpose_mask=True, p=1.0)
    ])


    class RITEDataset(Dataset):
        def __init__(self, imagenames, labelnames, masknames, wmasknames=None, labeldil5names=None, labeldil9names=None,
                     labeldil13names=None, transform=None):
            self.imagenames = imagenames
            self.labelnames = labelnames
            self.masknames = masknames
            self.wmasknames = wmasknames
            self.labeldil5names = labeldil5names
            self.labeldil9names = labeldil9names
            self.labeldil13names = labeldil13names
            self.transform = transform

        def __len__(self):
            return len(self.imagenames)

        def __getitem__(self, idx):
            image_resized = cv2.imread(self.imagenames[idx]).astype(np.uint8)[36:548, 26:538, :]
            label_resized = cv2.imread(self.labelnames[idx]).astype(np.float32)[36:548, 26:538, :] / 255
            mask_resized = cv2.imread(self.masknames[idx]).astype(np.float32)[36:548, 26:538, :] / 255
            if self.wmasknames is not None:
                wmask_resized = cv2.imread(self.wmasknames[idx]).astype(np.float32)[36:548, 26:538, :]
                label5dil_resized = cv2.imread(self.labeldil5names[idx]).astype(np.float32)[36:548, 26:538, :] / 255
                label9dil_resized = cv2.imread(self.labeldil9names[idx]).astype(np.float32)[36:548, 26:538, :] / 255
                label13dil_resized = cv2.imread(self.labeldil13names[idx]).astype(np.float32)[36:548, 26:538, :] / 255
                aug_data = self.transform(image=image_resized, masks=np.array(
                    [label_resized, mask_resized, wmask_resized, label5dil_resized, label9dil_resized, label13dil_resized]))
                x = aug_data['image'].float()
                y, m, wm, y5dil, y9dil, y13dil = aug_data['masks']
                y0 = y[:1, :, :] + y[1:2, :, :]
                y0 = (y0 == 1).float()
                y1 = y[:1, :, :] + y[1:2, :, :] + y[2:, :, :]
                y1 = (y1 > 0).float()
                y2 = y[2:, :, :] + y[1:2, :, :]
                y2 = (y2 == 1).float()
                y = torch.cat([y0, y1, y2], dim=0)
                return x, y, m.float(), wm.float(), y5dil.float(), y9dil.float(), y13dil.float()
            else:
                aug_data = self.transform(image=image_resized, masks=np.array([label_resized, mask_resized]))
                x = aug_data['image'].float()
                y, m = aug_data['masks']
                y0 = y[:1, :, :] + y[1:2, :, :]
                y0 = (y0 == 1).float()
                y1 = y[:1, :, :] + y[1:2, :, :] + y[2:, :, :]
                y1 = (y1 > 0).float()
                y2 = y[2:, :, :] + y[1:2, :, :]
                y2 = (y2 == 1).float()
                y = torch.cat([y0, y1, y2], dim=0)
                return x, y, m.float()


    def Crossentropy(y_pred, y_true, weight=None):
        y_pred = torch.clamp(y_pred, min=1e-6, max=1. - 1e-6)
        mse_ = -torch.special.xlogy(y_true, y_pred) - torch.special.xlogy(1 - y_true, 1 - y_pred)
        mse = torch.sum(mse_ * weight, dim=1)
        mse = torch.mean(mse, dim=[1, 2])
        return mse


    def Lossdice(y_pred, y_true, weight=None):
        dices = 1 - (2 * torch.sum(y_true * y_pred, dim=[2, 3]) + 1) / (
                torch.sum(y_true, dim=[2, 3]) + torch.sum(y_pred, dim=[2, 3]) + 1)
        return torch.sum(dices, dim=1)


    def MixLoss(y_pred, y_true, weight=None):
        # bce_loss = F.binary_cross_entropy(y_pred,y_true, weight=weight)
        bce_loss = Crossentropy(y_pred, y_true, weight=weight)
        dicel = Lossdice(y_pred, y_true)
        return torch.mean(bce_loss + 0.1 * dicel)


    def MixLoss2(y_pred, y_true, weight=None):
        bce_loss = Crossentropy(y_pred, y_true, weight=weight)
        dicel = Lossdice(y_pred, y_true)
        return torch.mean(bce_loss + dicel)


    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    unetn = results_config.nunet
    model = ConnectUnet(unetn, results_config.base_filters)
    #summary_model = summary(model, input_size=(1, 3, 512, 512))
    # print(summary_model)
    #logging.info(summary_model)

    # model_pra = reversed(model.state_dict().items())
    # next(model_pra)
    # print(next(model_pra)[-1].detach().to(device='cpu').numpy()[:,:,0,0].T)
    model.load_state_dict(torch.load('./%d/pysave/model_d%d_c%d.pth' % (flod, unetn, results_config.base_filters), weights_only=True))

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of trainable parameters: {num_params}')

    model = model.to(device=device)

    opt = optim.AdamW(model.parameters(), lr=0.002)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 10, T_mult=2, eta_min=0.0002)

    test_ds = RITEDataset(test_imagenames, test_labelnames, test_masknames, transform=transform_v)
    test_dl = DataLoader(
        test_ds,
        batch_size=results_config.batch_size,
        shuffle=False,
        drop_last=False
    )

    # model.load_state_dict(torch.load('./pysave/model.pth'))
    featuremapsavepath = './rite_result/featuremap_d%d_c%d' % (unetn, results_config.base_filters)
    if not os.path.exists(featuremapsavepath):
        os.makedirs(featuremapsavepath)

    predsavepath = './rite_result/pred_image_d%d_c%d' % (unetn, results_config.base_filters)
    if not os.path.exists(predsavepath):
        os.makedirs(predsavepath)

    ys = []
    yps = []
    ms = []
    with torch.no_grad():
        for nums, (x, y, m) in enumerate(test_dl):
            y_pred = 0
            for roti in range(1, 5, 1):
                x = x.to(device=device)
                y = y.to(device=device)
                m = m.to(device=device)
                x0in = torch.rot90(x, k=roti, dims=(2, 3))
                stime = time.time()
                y_predrot1, mseg = model(x0in)
                etime = time.time()
                print(etime - stime)

                np.save(featuremapsavepath + '/%d.npy' % nums, mseg.detach().to(device='cpu').numpy())
                y_predrot = y_predrot1[-1]  # *m
                y_predrot = torch.rot90(y_predrot, k=4 - roti, dims=(2, 3)).detach().to(device='cpu').numpy()
                for batchi in range(y.shape[0]):
                    np.save(predsavepath + '/%d.npy' % (y.shape[0] * nums + batchi),
                            y_predrot[batchi, :, :, :].transpose(1, 2, 0))
                y_pred += y_predrot  # *m.detach().to(device='cpu').numpy()
            y_pred /= 4

            y_pred = y_pred * m.detach().to(device='cpu').numpy()
            y = y.detach().to(device='cpu').numpy()
            # cv2.imwrite('RITE/result/%d.png'%nums,np.array(y_pred[0,:,:,:].transpose(1,2,0)>0.6, dtype=np.uint8)*255)
            # cv2.imwrite('RITE/result_true/%d.png'%nums,np.array(y[0,:,:,:].transpose(1,2,0)*255, dtype=np.uint8))
            m = m.detach().to(device='cpu').numpy()
            for batchi in range(y.shape[0]):
                # np.save('./pred_image/%d.npy'%(y.shape[0]*nums+batchi),y_pred[batchi,:,:,:].transpose(1,2,0))
                ys.append(y[batchi:(batchi + 1), :, :, :].transpose(0, 2, 3, 1))
                yps.append(y_pred[batchi:(batchi + 1), :, :, :].transpose(0, 2, 3, 1))
                ms.append(m[batchi:(batchi + 1), :, :, :].transpose(0, 2, 3, 1))
        from per_image_metrics import compute_per_image_metrics
        per_img = compute_per_image_metrics(yps, ys, ms)
        per_img_dir = './rite_result/per_image/d%d_c%d/fold%d' % (
            unetn, results_config.base_filters, flod)
        os.makedirs(per_img_dir, exist_ok=True)
        np.savez(os.path.join(per_img_dir, 'metrics.npz'), **per_img)
        # 也保留原始 npy 方便复现
        #np.save(os.path.join(per_img_dir, 'y_p.npy'), np.array(yps))
        #np.save(os.path.join(per_img_dir, 'y_t.npy'), np.array(ys))
        #np.save(os.path.join(per_img_dir, 'mask.npy'), np.array(ms))
        # ====================================

        # 原有的三行可以删掉：
        np.save('y_p2.npy', np.array(yps))
        np.save('y_t2.npy', np.array(ys))
        np.save('mask2.npy', np.array(ms))
    ind = evaluatecal(filename='_d%d_c%d_%d' % (unetn, results_config.base_filters, flod))
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
    