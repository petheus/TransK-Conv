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
import configs
from scipy.ndimage.interpolation import zoom

from indCal import evaluatecal

logging.basicConfig(filename=configs.timnow+'.log', filemode='a+', format='%(message)s', level=logging.INFO)

if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(torch.cuda.get_device_name(0))
else:
    print("cpu")

if not os.path.exists('./result'):
    os.makedirs('./result')

#path = '/home/ultramank/paper3/acdcd/data/ACDC'
path=configs.data_path
trainimages = ["patient{:0>3}".format(i) for i in [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
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


transform_t = A.Compose([
    #A.Resize(256,256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    #A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, p=0.5),
    ToTensorV2(transpose_mask=True, p=1.0)
])
transform_v = A.Compose([
    #A.Resize(256,256),
    ToTensorV2(transpose_mask=True, p=1.0)
])

class ACDCDataset(Dataset):
    def __init__(self, imagenames=train_imagenames, split="train", transform=None):
        self.imagenames = imagenames
        self.split = split
        self.transform = transform
        self.output_size=[256, 256]

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
            image = np.expand_dims(image,-1)
            label = np.expand_dims(label,-1)
            
            aug_data = self.transform(image=image, mask=label)
            image = aug_data['image'].float()
            label = torch.permute(F.one_hot(aug_data['mask'].long()[0,:,:], 4).float(), (2,0,1))
            return image, label, case
        
        else:
            image = h5f['image'][:]
            
            label = h5f['label'][:]
            _, x, y = image.shape
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
            #label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            image = np.expand_dims(image,1)
            #label = np.expand_dims(label,1)
            #label = torch.permute(F.one_hot(label, 4), (0,3,1,2))
            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).float()
            return image, label, case


def Crossentropy(y_pred, y_true, weight=None):
    y_pred = torch.clamp(y_pred, min=1e-6, max=1.-1e-6)
    mse_ = -torch.special.xlogy(y_true, y_pred)# - torch.special.xlogy(1-y_true, 1-y_pred)
    mse = torch.sum(mse_, dim=1)
    mse = torch.mean(mse, dim=1)
    mse = torch.mean(mse, dim=1)
    return mse


def Lossdice(y_pred, y_true, weight=None):
    dices = 1 - (2 * torch.sum(y_true * y_pred, dim=(2, 3)) + 1) / (
            torch.sum(y_true, dim=(2, 3)) + torch.sum(y_pred, dim=(2, 3)) + 1)
    return torch.sum(dices, dim=1)


def MixLoss(y_pred, y_true, weight=None):
    #bce_loss = F.binary_cross_entropy(y_pred,y_true, weight=weight)
    bce_loss = Crossentropy(y_pred,y_true, weight=weight)
    dicel = Lossdice(y_pred, y_true)
    return torch.mean(bce_loss+0.1*dicel)

def MixLoss2(y_pred, y_true, weight=None):
    bce_loss = Crossentropy(y_pred,y_true, weight=weight)
    dicel = Lossdice(y_pred, y_true)
    return torch.mean(bce_loss+dicel)

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
unetn=configs.nunet
model = ConnectUnet(unetn, configs.base_filters)
summary_model = summary(model, input_size=(1, 1, 256,256))
#print(summary(model, input_size=(1, 3, 256,256)))
logging.info(summary_model)
#model_pra = reversed(model.state_dict().items())
#next(model_pra)
#print(next(model_pra)[-1].detach().to(device='cpu').numpy()[:,:,0,0].T)
training = configs.training
if not training:
    model.load_state_dict(torch.load('./pysave/model_d%d_c%d.pth'%(unetn,configs.base_filters), weights_only=True))
model = model.to(device=device)
opt = optim.AdamW(model.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 10, T_mult=2, eta_min=0.0002)
train_ds = ACDCDataset(train_imagenames, "train", transform=transform_t)
train_dl = DataLoader(
            train_ds,
            batch_size=configs.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True
            )
test_ds = ACDCDataset(test_imagenames, "test", transform=transform_v)
print(len(train_imagenames))
for nums, (x, y, nameout) in enumerate(train_dl):
    print(x.shape)
    print(y.shape)
    yy = y.numpy()
    yy = np.reshape(yy, [-1])
    yy = yy.tolist()
    break
    


if training:
    for epoch in range(200):
        loss_sum = 0.
        start_time = time.time()
        model.train()
        for nums, (x, y, nameout) in enumerate(train_dl):
            x = x.to(device=device)
            y = y.to(device=device)
            y_pred, mseg = model(x)
            loss = MixLoss(y_pred=y_pred[-1], y_true=y)
            if unetn>1:
                loss += 0.1*MixLoss2(y_pred=y_pred[0], y_true=y)
            if unetn>2:
                loss += 0.1*MixLoss2(y_pred=y_pred[1], y_true=y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            #scheduler.step(epoch + nums / 20)
            loss_sum += loss.item()
        if (epoch+1)%1==0:
            torch.save(model.state_dict(), './pysave/model_d%d_c%d.pth'%(unetn,configs.base_filters))
        
        mtime = time.time()
        print('Epoch:%-4d'%epoch, 'Count:%d'%(nums+1), 'Train_loss: %.2f'%(loss_sum/(nums+1)), "Time: %.2f"%(mtime - start_time), end = ' ')
        logging_info = 'Epoch: %-4d Train_loss: %.2f Time: %.2f '%(epoch,(loss_sum/(nums+1)),(mtime - start_time))
        
        print('')
        if (epoch+1)%20==0 and epoch>=1:
            model.eval()
            for nums in range(len(test_imagenames)):
                x, y, nameout = test_ds[nums]
                y_pred = 0
                for roti in range(1,5,1):
                    x = x.to(device=device)
                    x0in = torch.rot90(x, k=roti, dims=(2,3))
                    with torch.no_grad():
                        y_predrot1, mseg = model(x0in)
                    y_predrot = y_predrot1[-1]
                    y_predrot = torch.rot90(y_predrot, k=4-roti, dims=(2,3)).detach().to(device='cpu')#.numpy()
                    y_pred += y_predrot#*m.detach().to(device='cpu').numpy()
                y_pred /= 4
                y_pred = F.interpolate(y_pred, size=[y.shape[-2], y.shape[-1]], mode='bilinear').numpy()
                y = y.numpy()
                y_preds=np.reshape(np.argmax(y_pred, axis=1).astype(np.uint8), [-1])
                y_trues = np.reshape(y, [-1])
                yc = [y_preds, y_trues.astype(np.uint8)]
                np.save('./result/'+nameout+'.npy', yc)
            metr = evaluatecal()
            print(metr, np.mean(metr, axis=0))
            logging.info(metr)
            logging.info(np.mean(metr, axis=0))
featuremapsavepath = './acdc_result/featuremap_d%d_c%d'%(unetn,configs.base_filters)
if not os.path.exists(featuremapsavepath):
    os.makedirs(featuremapsavepath)
    
predsavepath = './acdc_result/pred_image_d%d_c%d'%(unetn,configs.base_filters)
if not os.path.exists(predsavepath):
    os.makedirs(predsavepath)
       
#model.load_state_dict(torch.load('./pysave/model.pth'))
with torch.no_grad():
    for nums in range(len(test_imagenames)):
        x, y, nameout = test_ds[nums]
        np.save('./label/%d.npy'%nums, y)
        np.save('./image/%d.npy'%nums, x)
        print(nameout)
        y_pred = 0
        for roti in range(1,5,1):
            x = x.to(device=device)
            x0in = torch.rot90(x, k=roti, dims=(2,3))
            y_predrot1, mseg = model(x0in)
            np.save(featuremapsavepath+'/%d.npy'%nums, mseg.detach().to(device='cpu').numpy())
            y_predrot = y_predrot1[-1]#*m
            y_predrot = torch.rot90(y_predrot, k=4-roti, dims=(2,3)).detach().to(device='cpu')#.numpy()

            y_pred += y_predrot#*m.detach().to(device='cpu').numpy()
        y_pred /= 4
        y_pred = F.interpolate(y_pred, size=[y.shape[-2], y.shape[-1]], mode='bilinear').numpy()
        #cv2.imwrite('RITE/result/%d.png'%nums,np.array(y_pred[0,:,:,:].transpose(1,2,0)>0.6, dtype=np.uint8)*255)
        #cv2.imwrite('RITE/result_true/%d.png'%nums,np.array(y[0,:,:,:].transpose(1,2,0)*255, dtype=np.uint8))
        y = y.numpy()
        y_preds=np.reshape(np.argmax(y_pred, axis=1).astype(np.uint8), [-1])
        y_trues = np.reshape(y, [-1])
        yc = [y_preds, y_trues.astype(np.uint8)]
        np.save('./result/'+nameout+'.npy', yc)
evaluatecal(if_result=True, filename='_d%d_c%d'%(unetn,configs.base_filters))
