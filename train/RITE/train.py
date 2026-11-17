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
#from pynet import ConnectUnet
from connectunets import ConnectUnet
#from nunetr import ConnectUnet
from sklearn.metrics import roc_auc_score
import logging
from torchinfo import summary
import configs

from ptflops import get_model_complexity_info

'''timnow = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
filename = "./history/"+timnow
os.system('mkdir '+filename)
os.system('cp *.py '+filename)
os.system('mv *.log ./history/')
with open(timnow+'.log', 'w+') as f:
    pass
logging.basicConfig(filename=timnow+'.log', filemode='w+', format='%(message)s', level=logging.INFO)'''
logging.basicConfig(filename=configs.timnow+'.log', filemode='a+', format='%(message)s', level=logging.INFO)
from indCal import evaluatecal

if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(torch.cuda.get_device_name(0))
else:
    print("cpu")

flod = configs.flod
path = configs.data_path#'./RITE'
np.random.seed(10)
num_ = np.arange(20)
np.random.shuffle(num_)
num_ = num_
trainimages = os.listdir(path + '/training/images_png')
trainimages = list(filter(lambda x: x[-3:]=='png', trainimages))
trainimages = np.array(list(trainimages))
trainimages = trainimages[num_]
testimages = os.listdir(path + '/test/images_png')
testimages = list(filter(lambda x: x[-3:]=='png', testimages))
testimages.sort()
print(testimages)
train_imagenames = [path + '/training/images_png/' + name for name in trainimages]
test_imagenames = [path + '/test/images_png/' + name for name in testimages]
train_labelnames = [path + '/training/av/' + name for name in trainimages]

test_labelnames = [path + '/test/av/' + name for name in testimages]
train_masknames = [path + '/training/mask_png/' + name for name in trainimages]
test_masknames = [path + '/test/mask_png/' + name for name in testimages]
train_wmasknames = [path + '/training/weight_label/' + name for name in trainimages]

transform_t = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, p=0.5),
    A.Normalize(
        mean=[0.]*3,
        std=[1.]*3,
        max_pixel_value=1.0
    ),
    ToTensorV2(transpose_mask=True, p=1.0)
])
transform_v = A.Compose([
    A.Normalize(
        mean=[0.]*3,
        std=[1.]*3,
        max_pixel_value=1.0
    ),
    ToTensorV2(transpose_mask=True, p=1.0)
])

class RITEDataset(Dataset):
    def __init__(self, imagenames, labelnames, masknames, wmasknames=None, transform=None):
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
        image_resized = cv2.imread(self.imagenames[idx]).astype(np.uint8)[36:548,26:538,:]
        label_resized = cv2.imread(self.labelnames[idx]).astype(np.float32)[36:548,26:538,:]/255
        mask_resized = cv2.imread(self.masknames[idx]).astype(np.float32)[36:548,26:538,:]/255
        if self.wmasknames is not None:
            wmask_resized = cv2.imread(self.wmasknames[idx]).astype(np.float32)[36:548,26:538,:]
            aug_data = self.transform(image=image_resized, masks=np.array([label_resized, mask_resized, wmask_resized]))
            x = aug_data['image'].float()
            y, m, wm = aug_data['masks']
            y0 = y[:1, :, :] + y[1:2, :, :]
            y0 = (y0 == 1).float()
            y1 = y[:1, :, :] + y[1:2, :, :] + y[2:, :, :]
            y1 = (y1 > 0).float()
            y2 = y[2:, :, :] + y[1:2, :, :]
            y2 = (y2 == 1).float()
            y = torch.cat([y0, y1, y2], dim=0)
            return x, y, m.float(), wm.float()
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
    y_pred = torch.clamp(y_pred, min=1e-6, max=1.-1e-6)
    mse_ = -torch.special.xlogy(y_true, y_pred) - torch.special.xlogy(1-y_true, 1-y_pred)
    mse = torch.sum(mse_ * weight, dim=1)
    mse = torch.mean(mse, dim=[1, 2])
    return mse


def Lossdice(y_pred, y_true, weight=None):
    dices = 1 - (2 * torch.sum(y_true * y_pred, dim=[2, 3]) + 1) / (
            torch.sum(y_true, dim=[2, 3]) + torch.sum(y_pred, dim=[2, 3]) + 1)
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
summary_model = summary(model, input_size=(1, 3, 512,512))
#print(summary_model)
logging.info(summary_model)



#model_pra = reversed(model.state_dict().items())
#next(model_pra)
#print(next(model_pra)[-1].detach().to(device='cpu').numpy()[:,:,0,0].T)
training = configs.training
if not training:
	model.load_state_dict(torch.load('./pysave/model_d%d_c%d.pth'%(unetn,configs.base_filters), weights_only=True))

num_params = sum(p.numel() for p in model.parameters())
print(f'Total number of trainable parameters: {num_params}')

flops, params = get_model_complexity_info(model, (3, 512,512), as_strings=True, print_per_layer_stat=True)
print('flops: ', flops, 'params: ', params)

model = model.to(device=device)



opt = optim.AdamW(model.parameters(), lr=0.002)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 10, T_mult=2, eta_min=0.0002)
train_ds = RITEDataset(train_imagenames, train_labelnames, train_masknames, train_wmasknames, transform=transform_t)
train_dl = DataLoader(
            train_ds,
            batch_size=configs.batch_size,
            shuffle=False,
            drop_last=False
            )
test_ds = RITEDataset(test_imagenames, test_labelnames, test_masknames, transform=transform_v)
test_dl = DataLoader(
    test_ds,
    batch_size=configs.batch_size,
    shuffle=False,
    drop_last=False
)

if training:
    for epoch in range(500):
        loss_sum = 0.
        start_time = time.time()
        model.train()
        for nums, (x, y, m, wm) in enumerate(train_dl):
            stime = time.time()
            x = x.to(device=device)
            y = y.to(device=device)
            m = m.to(device=device)
            wm = wm.to(device=device)
            
            y_pred, mseg = model(x)
            
            loss = MixLoss(y_pred=y_pred[-1]*m, y_true=y, weight=m*wm)
            if unetn>1:
                loss += 0.1*MixLoss2(y_pred=y_pred[0]*m, y_true=y, weight=m*wm)
            if unetn>2:
                loss += 0.1*MixLoss2(y_pred=y_pred[1]*m, y_true=y, weight=m*wm)
            opt.zero_grad()
            loss.backward()
            opt.step()
            #scheduler.step(epoch + nums / 20)
            loss_sum += loss.item()
            etime = time.time()
            #print(etime-stime)
        x = x.detach().to(device='cpu')
        y = y.detach().to(device='cpu')
        m = m.detach().to(device='cpu')
        wm = wm.detach().to(device='cpu')
        #y5dil = y5dil.detach().to(device='cpu')
        #y9dil = y9dil.detach().to(device='cpu')
        #y13dil = y13dil.detach().to(device='cpu')
        
        mtime = time.time()
        print('Epoch:%-4d'%epoch, 'Train_loss: %.2f'%(loss_sum/(nums+1)), "Time: %.2f"%(mtime - start_time), end = ' ')
        logging_info = 'Epoch: %-4d Train_loss: %.2f Time: %.2f '%(epoch,(loss_sum/(nums+1)),(mtime - start_time))
        if epoch<500 and epoch>450:
            start_time = time.time()
            y_preds = []
            y_trues = []
            masks = []
            model.eval()
            for nums, (x, y, m) in enumerate(test_dl):
                y_pred = 0
                x = x.to(device=device)
                y = y.to(device=device)
                m = m.to(device=device)
                for roti in range(1,5,1):
                    x0in = torch.rot90(x, k=roti, dims=(2,3))
                    #x0in = x0in.to(device=device)
                    with torch.no_grad():
                        y_predrot1, mseg = model(x0in)
                    y_predrot = y_predrot1[-1]#*m
                    y_predrot = torch.rot90(y_predrot.detach().to(device='cpu'), k=4-roti, dims=(2,3)).numpy()
                    y_pred += y_predrot#*m.detach().to(device='cpu').numpy()
                y_pred /= 4
                y_pred = y_pred*m.detach().to(device='cpu').numpy()
                y_preds.append(np.reshape(y_pred[:,::2,:,:].transpose(0,2,3,1), [-1,2]))
                y = y.detach().to(device='cpu').numpy()
                #y = y.numpy()
                y_trues.append(np.reshape(y[:,::2,:,:].transpose(0,2,3,1), [-1,2]))
                m = m.detach().to(device='cpu').numpy()
                #m = m.numpy()
                masks.append(np.reshape(m[:,::2,:,:], [-1]))
            mtime = time.time()
            y_preds = np.concatenate(y_preds)
            y_trues = np.concatenate(y_trues)
            rocauc = roc_auc_score(y_trues, y_preds)
            
            
            if 'best_result' in dir():
                if best_result<rocauc:
                    torch.save(model.state_dict(), './pysave/model_best_d%d_c%d.pth'%(unetn,configs.base_filters))
                    best_result=1*rocauc
                    print('Test auc%f'%rocauc, "Time: %.2f"%(mtime - start_time), '-best')
                    logging.info(logging_info+'Test auc%f Time: %.2f -best'%(rocauc,(mtime - start_time)))
                else:
                    print('Test auc%f'%rocauc, "Time: %.2f"%(mtime - start_time))
                    logging.info(logging_info+'Test auc%f Time: %.2f'%(rocauc,(mtime - start_time)))
            else:
                torch.save(model.state_dict(), './pysave/model_d%d_c%d.pth'%(unetn,configs.base_filters))
                best_result=1*rocauc
                print('Test auc%f'%rocauc, "Time: %.2f"%(mtime - start_time))
                logging.info(logging_info+'Test auc%f Time: %.2f'%(rocauc,(mtime - start_time)))
        else:
            print('')
        if (epoch+1)%50==0 and epoch>50:
            torch.save(model.state_dict(), './pysave/model_d%d_c%d.pth'%(unetn,configs.base_filters))
            ys = []
            yps = []
            ms = []
            model.eval()
            
            for nums, (x, y, m) in enumerate(test_dl):
                y_pred = 0
                for roti in range(1,5,1):
                    x = x.to(device=device)
                    y = y.to(device=device)
                    m = m.to(device=device)
                    x0in = torch.rot90(x, k=roti, dims=(2,3))
                    with torch.no_grad():
                        y_predrot1, mseg = model(x0in)
                    y_predrot = y_predrot1[-1]#*m
                    y_predrot = torch.rot90(y_predrot, k=4-roti, dims=(2,3)).detach().to(device='cpu').numpy()
                    y_pred += y_predrot#*m.detach().to(device='cpu').numpy()
                y_pred /= 4
                y_pred = y_pred*m.detach().to(device='cpu').numpy()
                y = y.detach().to(device='cpu').numpy()
                m = m.detach().to(device='cpu').numpy()
                for batchi in range(y.shape[0]):
                    ys.append(y[batchi:(batchi+1),:,:,:].transpose(0,2,3,1))
                    yps.append(y_pred[batchi:(batchi+1),:,:,:].transpose(0,2,3,1))
                    ms.append(m[batchi:(batchi+1),:,:,:].transpose(0,2,3,1))
            np.save('y_p2.npy',np.array(yps))
            np.save('y_t2.npy',np.array(ys))
            np.save('mask2.npy',np.array(ms))
            metrs0, metr = evaluatecal()
            logging.info(metrs0)
            logging.info(metr)
        
#model.load_state_dict(torch.load('./pysave/model.pth'))
featuremapsavepath = './rite_result/featuremap_d%d_c%d'%(unetn,configs.base_filters)
if not os.path.exists(featuremapsavepath):
    os.makedirs(featuremapsavepath)
    
predsavepath = './rite_result/pred_image_d%d_c%d'%(unetn,configs.base_filters)
if not os.path.exists(predsavepath):
    os.makedirs(predsavepath)

ys = []
yps = []
ms = []
with torch.no_grad():
    
    for nums, (x, y, m) in enumerate(test_dl):
        y_pred = 0
        for roti in range(1,5,1):
            x = x.to(device=device)
            y = y.to(device=device)
            m = m.to(device=device)
            x0in = torch.rot90(x, k=roti, dims=(2,3))
            stime = time.time()
            y_predrot1, mseg = model(x0in)
            etime = time.time()
            print(etime-stime)
            

            np.save(featuremapsavepath+'/%d.npy'%nums, mseg.detach().to(device='cpu').numpy())
            y_predrot = y_predrot1[-1]#*m
            y_predrot = torch.rot90(y_predrot, k=4-roti, dims=(2,3)).detach().to(device='cpu').numpy()
            for batchi in range(y.shape[0]):
                np.save(predsavepath+'/%d.npy'%(y.shape[0]*nums+batchi),y_predrot[batchi,:,:,:].transpose(1,2,0))
            y_pred += y_predrot#*m.detach().to(device='cpu').numpy()
        y_pred /= 4
        
        y_pred = y_pred*m.detach().to(device='cpu').numpy()
        y = y.detach().to(device='cpu').numpy()
        #cv2.imwrite('RITE/result/%d.png'%nums,np.array(y_pred[0,:,:,:].transpose(1,2,0)>0.6, dtype=np.uint8)*255)
        #cv2.imwrite('RITE/result_true/%d.png'%nums,np.array(y[0,:,:,:].transpose(1,2,0)*255, dtype=np.uint8))
        m = m.detach().to(device='cpu').numpy()
        for batchi in range(y.shape[0]):
            #np.save('./pred_image/%d.npy'%(y.shape[0]*nums+batchi),y_pred[batchi,:,:,:].transpose(1,2,0))
            ys.append(y[batchi:(batchi+1),:,:,:].transpose(0,2,3,1))
            yps.append(y_pred[batchi:(batchi+1),:,:,:].transpose(0,2,3,1))
            ms.append(m[batchi:(batchi+1),:,:,:].transpose(0,2,3,1))
    np.save('y_p2.npy',np.array(yps))
    np.save('y_t2.npy',np.array(ys))
    np.save('mask2.npy',np.array(ms))
evaluatecal(if_result=True, filename='_d%d_c%d'%(unetn,configs.base_filters))
