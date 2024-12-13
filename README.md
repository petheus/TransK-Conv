# TransK-Conv: Convolution based on kernel recomposed by transformer for artery vein classification in fundus images
This repository is for the paper "TransK-Conv: Convolution based on kernel recomposed by transformer for artery vein classification in fundus images".
## Dataset
The RITE dataset can be obtained from [here](https://medicine.uiowa.edu/eye/rite-dataset).<br>
The HRF dataset can be obtained from [here](https://www5.cs.fau.de/research/data/fundus-images/)<br>
The HRF labels re-labeled by Chen et al. are available [here](https://github.com/o0t1ng0o/TW-GAN.git)<br>
*Extract all the data into a new folder named RITE or HRF in the root directory.*<br>
## Usage
Start by running two files in the preprocessing folder.<br>
```bash
python ./preprocessing/encode_to_png_RITE.py
python ./preprocessing/encode_to_png_HRF.py
```
Then
