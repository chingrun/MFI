Multiscale feature integration network for lithology identification from point cloud data
=
This is the implementation of Multiscale feature integration network on Python 3 and PyTorch.

How to use it
--
1. run the DDPM training, trained weights are stored in 'Checkpoints/' folder. 
```Python
python DDPM_train.py
```

2. run the DDPM-CR training, train cloud removal head with DDPM features from pretrained weights stored in 'Checkpoints/' folder.  
```Python
python DDPM-CR.py
```

Requirements
--
* torch 1.10.0+cu113
* torchvision 0.11.1+cu113
* gdal 3.0.2
* numpy 1.19.3
* scikit-learn 0.24.2
