# -*- coding: utf-8 -*-
"""
@author: wentingchen
Code for generate shuffled image for datasets.
Dataset:
AV-DRIVE - only shuffle the trainset
HRF - shuffle all the ground-truth
"""
import numpy as np
import cv2
import os
from twtool.genFakeSample import generate, filterGreenWhite
from tqdm import tqdm

shuffle_rate_range=[0.05,0.25]
dataset_root_list = './RITE/training/av'
save_root_list = './RITE/shuffled'

print("Generating shuffled data for AV-DRIVE and HRF dataset with shuffle rate rangine from " + str(shuffle_rate_range[0]) + " to " + str(shuffle_rate_range[1]))
if not os.path.exists(save_root_list):
    os.makedirs(save_root_list)
    print("Making directory: ", save_root_list)

input_av_root = dataset_root_list
output_av_root = save_root_list
print("Input data root :", input_av_root)
print("Output data root :", output_av_root)

imglist = os.listdir(input_av_root)
num = len(imglist)

for i in tqdm(range(num)):
    shuffle_ratio = np.random.uniform(shuffle_rate_range[0], shuffle_rate_range[1], 1)
    # read image
    AVSeg = cv2.imread(os.path.join(input_av_root, imglist[i]))
    AVSeg = cv2.cvtColor(AVSeg, cv2.COLOR_BGR2RGB)

    AVSeg = filterGreenWhite(AVSeg)
    fakeAVSeg = generate(AVSeg, shuffle_ratio=shuffle_ratio)
    fakeAVSeg = cv2.cvtColor(fakeAVSeg, cv2.COLOR_RGB2BGR)
    fakeAVSeg = np.uint8(fakeAVSeg)
    # save fake AVSeg image
    fakeAVSeg[:,:,1] = fakeAVSeg[:,:,0]+fakeAVSeg[:,:,2]
    cv2.imwrite(os.path.join(output_av_root, imglist[i]), fakeAVSeg)

print("---------------------------------------")
    