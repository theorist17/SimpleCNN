# ImageNet

Customized image classification network using PyTorch. 88% ACC on test.

The dataset I used consists of two classes, human and plants, collected from ImageNet.
The network I designed use the basic of CNN, batch normalization, RELU.
It supports simple hyper parmetric search for things like learning rate using validation dataset.

## Geting Started

### Prerequites

Things you need to install this software

```
CUDA
cuDNN
```

### Installing

Create a Python environment using conda or virtual env
Install packages from requirements.txt

```
pip install -r requirements.txt
```
or
```
conda env update --file requirements.txt
```
## Training

It will train MyNet to classify between two classes; human and plants.

```
python train.py
```

Before running, make sure you set DOWNLOAD_DATASET = True, so you can download the some portion of dataset that I selected from ImageNet.
Once you run the train.py, it will automatically get urls of those, then download the actual images into /data.
It will split dataset into train, validation, and test, and build numpy serialized dataset.

## Logging
Loss and accuracy for every epoch in the training will be logged into /log.
Important for hyperparmeter optimization.
![Alt text](assets/MyNet5Loss.png?raw=true "Title")
![Alt text](assets/MyNet5Acc.png?raw=true "Title")

## Directory Sturcture
The trained models will be saved into /models
![Alt text](assets/directory.png?raw=true "Title")

## Testing
The test accuracy is 0.8869. Not so impressive, but the code is simple enough for the AI novice to see the whole CNN buildup.
![Alt text](assets/test.png?raw=true "Title")
## Authors
* **Hong-In Lee** - *SimpleCNN* - [theorist17](https://github.com/theorist17)

## Acknowledgements 

* [ImageNet - Sysnet](http://image-net.org/synset?wnid=n02084071) - The ImageNet is under maintence for months, as of June 2020. But you can use exploration dataset.
* [CS231n: Convolutional Neural Networks for Visual Recognition](https://www.youtube.com/watch?v=vT1JzLTH4G4) - A good start for you to learn CNN. (The professor basically made ImageNet)
* [TRAINING A CLASSIFIER](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) - The official PyTorch tutorial for building CNN on ImageNet.
 
