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
import configs

from indCal import evaluatecal

logging.basicConfig(filename=configs.timnow+'.log', filemode='a+', format='%(message)s', level=logging.INFO)

if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(torch.cuda.get_device_name(0))
else:
    print("cpu")

if not os.path.exists('./fmname'):
    os.makedirs('./fmname')
if not os.path.exists('./result'):
    os.makedirs('./result')
path =configs.data_path
if not os.path.exists('./num_rand.npy'):
    np.random.seed(10)
    num_ = np.arange(2594)
    np.random.shuffle(num_)
    np.save('./num_rand.npy', num_)
else:
    num_ = np.load('./num_rand.npy')
if (not os.path.exists('./trainimages.npy')) or (not os.path.exists('./testimages.npy')):
    trainimagesa = os.listdir(path + '/ISIC2018_Task1-2_Training_Input_png')
    trainimagesa = list(filter(lambda x: x[-3:]=='png', trainimagesa))
    trainimagesa =  np.array(list(trainimagesa))[num_]
    trainimages =trainimagesa[:2075]
    np.save('./trainimages.npy', trainimages)
    testimages = trainimagesa[2075:]
    np.save('./testimages.npy', testimages)
else:
    trainimages = np.load('./trainimages.npy')
    testimages = np.load('./testimages.npy')

#testimages = os.listdir(path + '/ISIC2018_Task1-2_Test_Input')
#testimages = list(filter(lambda x: x[-3:]=='jpg', testimages))
train_imagenames = [path + '/ISIC2018_Task1-2_Training_Input_png/' + name for name in trainimages]
test_imagenames = [path + '/ISIC2018_Task1-2_Training_Input_png/' + name for name in testimages]
train_labelnames = [path + '/ISIC2018_Task1_Training_GroundTruth_png/' + name[:-4]+'_segmentation.png' for name in trainimages]
test_labelnames = [path + '/ISIC2018_Task1_Training_GroundTruth_png/' + name[:-4]+'_segmentation.png' for name in testimages]

transform_t = A.Compose([
    #A.Resize(256,256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    #A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, p=0.5),
    A.Normalize(
        mean=[0.]*3,
        std=[1.]*3,
        max_pixel_value=255.0
    ),
    ToTensorV2(transpose_mask=True, p=1.0)
])
transform_v = A.Compose([
    #A.Resize(256,256),
    A.Normalize(
        mean=[0.]*3,
        std=[1.]*3,
        max_pixel_value=255.0
    ),
    ToTensorV2(transpose_mask=True, p=1.0)
])

class ISICDataset(Dataset):
    def __init__(self, imagenames=train_imagenames, labelnames=train_labelnames, transform=transform_t):
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
        y = aug_data['mask'].float()/255.
        nameout = self.imagenames[idx].split('/')[-1][:-4]
        #y = y.unsqueeze(0)
        return x, y, nameout


def Crossentropy(y_pred, y_true, weight=None):
    y_pred = torch.clamp(y_pred, min=1e-6, max=1.-1e-6)
    mse_ = -torch.special.xlogy(y_true, y_pred) - torch.special.xlogy(1-y_true, 1-y_pred)
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
summary_model = summary(model, input_size=(1, 3, 256,256))
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
train_ds = ISICDataset(train_imagenames, train_labelnames, transform=transform_t)
train_dl = DataLoader(
            train_ds,
            batch_size=configs.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True
            )
test_ds = ISICDataset(test_imagenames, test_labelnames, transform=transform_v)
test_dl = DataLoader(
    test_ds,
    batch_size=configs.batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=True
)

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
        if (epoch+1)%50==0:
            torch.save(model.state_dict(), './pysave/model_d%d_c%d.pth'%(unetn,configs.base_filters))
        
        mtime = time.time()
        print('Epoch:%-4d'%epoch, 'Count:%d'%(nums+1), 'Train_loss: %.2f'%(loss_sum/(nums+1)), "Time: %.2f"%(mtime - start_time), end = ' ')
        logging_info = 'Epoch: %-4d Train_loss: %.2f Time: %.2f '%(epoch,(loss_sum/(nums+1)),(mtime - start_time))
        if epoch<0 or epoch>20000:
            start_time = time.time()
            model.eval()
            rocauc = 0.
            inums = 0
            for nums, (x, y, nameout) in enumerate(test_dl):
                y_pred = 0
                x = x.to(device=device)
                for roti in range(1,5,1):
                    x0in = torch.rot90(x, k=roti, dims=(2,3))
                    #x0in = x0in.to(device=device)
                    with torch.no_grad():
                        y_predrot1, mseg = model(x0in)
                    y_predrot = y_predrot1[-1]#*m
                    y_predrot = torch.rot90(y_predrot.detach().to(device='cpu'), k=4-roti, dims=(2,3))#.numpy()
                    y_pred += y_predrot#*m.detach().to(device='cpu').numpy()
                y_pred /= 4
                y_pred = F.interpolate(y_pred, size=[y.shape[-2], y.shape[-1]], mode='bilinear').numpy()
                y = y.numpy()
                for batchi in range(y.shape[0]):
                    y_preds=np.reshape(y_pred[batchi,0,:,:], [-1,1])
                    y_trues = np.reshape(y[batchi,0,:,:], [-1,1])
                    if np.min(y_trues)==1:
                        pass
                    else:
                        rocauc += roc_auc_score(y_trues, y_preds)
                        inums += 1
                    yc = [y_preds[:,0]>0.5, y_trues[:,0]>0.5]
                    np.save('./result/'+nameout[batchi]+'.npy', yc)
            mtime = time.time()
            rocauc/=inums
            
            
            if 'best_result' in dir():
                if best_result<rocauc:
                    torch.save(model.state_dict(), './pysave/model_best_d%d_c%d.pth'%(unetn,configs.base_filters))
                    best_result=1*rocauc
                    print('Test auc%f'%rocauc, "Time: %.2f"%(mtime - start_time), '-best')
                    logging.info(logging_info+'Test auc%f Time: %.2f -best'%(rocauc,(mtime - start_time)))
                else:
                    torch.save(model.state_dict(), './pysave/model_d%d_c%d.pth'%(unetn,configs.base_filters))
                    print('Test auc%f'%rocauc, "Time: %.2f"%(mtime - start_time))
                    logging.info(logging_info+'Test auc%f Time: %.2f'%(rocauc,(mtime - start_time)))
            else:
                torch.save(model.state_dict(), './pysave/model_d%d_c%d.pth'%(unetn,configs.base_filters))
                best_result=1*rocauc
                print('Test auc%f'%rocauc, "Time: %.2f"%(mtime - start_time))
                logging.info(logging_info+'Test auc%f Time: %.2f'%(rocauc,(mtime - start_time)))
        else:
            print('')
        if (epoch+1)%20==0 and epoch>=1:
            model.eval()
            
            for nums, (x, y, nameout) in enumerate(test_dl):
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
                for batchi in range(y.shape[0]):
                    y_preds=np.reshape(y_pred[batchi,0,:,:], [-1])
                    y_trues = np.reshape(y[batchi,0,:,:], [-1])
                    yc = [y_preds>0.5, y_trues>0.5]
                    np.save('./result/'+nameout[batchi]+'.npy', yc)
            metr = evaluatecal()
            logging.info(metr)
  
torch.save(model.state_dict(), './pysave/model_d%d_c%d.pth'%(unetn,configs.base_filters))
#model.load_state_dict(torch.load('./pysave/model_d%d_c%d.pth'%(unetn,configs.base_filters)))
featuremapsavepath = './isic_result/featuremap_d%d_c%d'%(unetn,configs.base_filters)
if not os.path.exists(featuremapsavepath):
    os.makedirs(featuremapsavepath)
    
predsavepath = './isic_result/pred_image_d%d_c%d'%(unetn,configs.base_filters)
if not os.path.exists(predsavepath):
    os.makedirs(predsavepath)
with torch.no_grad():
    for nums, (x, y, nameout) in enumerate(test_dl):
        np.save('./fmname/%d.npy'%nums, nameout)
        y_pred = 0
        for roti in range(1,5,1):
            x = x.to(device=device)
            x0in = torch.rot90(x, k=roti, dims=(2,3))
            y_predrot1, mseg = model(x0in)
            np.save(featuremapsavepath+'/%d.npy'%nums, mseg.detach().to(device='cpu').numpy())
            y_predrot = y_predrot1[-1]#*m
            y_predrot = torch.rot90(y_predrot, k=4-roti, dims=(2,3)).detach().to(device='cpu')#.numpy()
            for batchi in range(y.shape[0]):
                np.save(predsavepath+'/%d.npy'%(y.shape[0]*nums+batchi),y_predrot[batchi,:,:,:].numpy()
.transpose(1,2,0))
            y_pred += y_predrot#*m.detach().to(device='cpu').numpy()
        y_pred /= 4
        y_pred = F.interpolate(y_pred, size=[y.shape[-2], y.shape[-1]], mode='bilinear').numpy()
        #cv2.imwrite('RITE/result/%d.png'%nums,np.array(y_pred[0,:,:,:].transpose(1,2,0)>0.6, dtype=np.uint8)*255)
        #cv2.imwrite('RITE/result_true/%d.png'%nums,np.array(y[0,:,:,:].transpose(1,2,0)*255, dtype=np.uint8))
        y = y.numpy()
        for batchi in range(y.shape[0]):
            y_preds=np.reshape(y_pred[batchi,0,:,:], [-1])
            y_trues = np.reshape(y[batchi,0,:,:], [-1])
            yc = [y_preds>0.5, y_trues>0.5]
            np.save('./result/'+nameout[batchi]+'.npy', yc)
evaluatecal(if_result=True, filename='_d%d_c%d'%(unetn,configs.base_filters))
