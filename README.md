# TransK-Conv: Context convolution based on kernel reconstructed by transformer for medical image segmentation
This repository is for the paper "TransK-Conv: Context convolution based on kernel reconstructed by transformer for medical image segmentation".
## Getting Started
```
#Train the RITE dataset. The dataset path is set in train/RITE/configs.py.
cd train/RITE
sh run.sh
#Train the ISIC dataset. The dataset path is set in train/ISIC/configs.py.
cd train/ISIC
sh run.sh
#Train the ACDC dataset. The dataset path is set in train/ACDC/configs.py.
cd train/ACDC
sh run.sh
```
The slices files of ACDC we used in the experiment were the processed ones provided by the original repository of [ScribFormer](https://github.com/duweidai/Ms-RED.git).
