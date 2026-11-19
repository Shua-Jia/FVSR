


# Multiscale Face Video Super-Resolution with Bounded Timestep Reweighting
This is the program page of our latest approach for Face Video Super-Resolution(FVSR).  


### Environment

The proposed framework is implemented in Python using PyTorch 2.4.1 and Torchvision 0.19.1, both with CUDA 12.1 support. The software environment includes NumPy 1.24.4, SciPy 1.10.1, Pillow 9.1.0, imgaug 0.4.0, scikit-image 0.21.0, and LPIPS 0.1.4.

### Datasets
This study uses publicly available datasets. The CelebV-HQ used for training and testing is available at https://drive.google.com/drive/folders/19DLr27P9xMOTn_W6hxpxxm8_5jJoX-nR.

VFHQ also used for tesing is available at https://liangbinxie.github.io/projects/vfhq/



### Eval
After the training you can run the following command to FVSR for evaluation. 

python eval.py --data_path 




