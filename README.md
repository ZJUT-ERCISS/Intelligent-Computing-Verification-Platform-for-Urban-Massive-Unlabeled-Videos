# Meta-learning_Metric_System

## Introduction

The meta-learning metric system based on weak correlation semantic features is a context-aware system that can quickly locate pedestrian events and understand semantic content in real time. It has a wide range of applications in a small amount of annotated urban surveillance scenes, active perception of massive data and other fields.

The prototype representation of the category can be formed by only a small number of samples. By measuring the feature distance between the scene video feature and the prototype feature, the semantic category of the unseen scene can be recognized, which significantly improves the semantic detection ability in the actual scene.

## Dataset

In the general action recognition task, a data set containing 144 categories and a total of 156910 samples is constructed for system training and validation. It contains 120 categories of basic motion data from the "NTU RGB+D 120" dataset, totaling 114,480 3D skeleton data. In addition, in order to enhance the two-dimensional skeleton perception ability of the model, this system extracts 24 representative behavior categories from the kinetic data set, totaling 42430 video samples, and extracts two-dimensional skeleton information from them to form the data set content.

In terms of single-sample action recognition tasks, this system constructs a single-sample action recognition dataset. Following the rules of single-sample action recognition dataset, the training set is composed of 100 categories of data in the "NTU RGB+D 120" dataset. The test set consists of another 20 categories from the "NTU RGB+D 120" dataset and the multi-camera fall behavior scene video dataset MCFD. In the test set, this system takes one sample of each category as the reference sample and the other samples as the test samples.

In the unsupervised behavioral semantic temporal localization task, the proposed system randomly selects 176 samples from the ActivityNet 1.3 dataset as the test set.

## Preparation

System：Linux Ubuntu20.04

CUDA Version: 11.4

GPU: NVIDIA 3090

Pytorch-gpu: 1.12.0

You can download our resource files [here](https://zjuteducn-my.sharepoint.com/personal/211122120051_zjut_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F211122120051%5Fzjut%5Fedu%5Fcn%2FDocuments%2Fresources&ga=1).

## Training

You can use the following command to train model:
```
python ./train_model.py
```


## Inference

First, you need to create a list file that contains the video file paths and categories as an argument to your inference script. An example is followed:

```
video/example.mp4 42
```

Then you can use the following command to infer a video:

```
python ./engine/infer_action.py
```


