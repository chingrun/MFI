Multiscale feature integration network for lithology identification from point cloud data
=
This is the implementation of Multiscale feature integration network on Python 3 and PyTorch.

How to use it
--
1. run the MFI network training, with specified train file and test file paths, trained weights are stored at 'checkpoints/' folder in default. 
```Python
python train.py
```

2. run the inference, with specified txt file paths on inference, pretrained weight path and saved inference file path.  
```Python
python inference.py
```

Requirements
--
* torch 1.10.0+cu113
* torchvision 0.11.1+cu113
* pandas 1.1.5
* numpy 1.19.3
* scikit-learn 0.24.2
