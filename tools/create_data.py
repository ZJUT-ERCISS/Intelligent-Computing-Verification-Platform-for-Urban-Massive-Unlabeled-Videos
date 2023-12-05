import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as F
import os
import sys
import time
import copy
import json
import torch
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter

from terminaltables import AsciiTable
import random
from torch.autograd import Variable
from scipy.interpolate import interp1d

from mmcv import Config, load
from mmcv.runner import load_checkpoint
from datasets.builder import build_dataloader, build_dataset

device = torch.device("cuda:1" )
# os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID
# model_file = '/home/shy/code_hik/video_model/.vscode/train/ntu_100/best_top1_acc_epoch_26.pth'
model_file = '/home/code/video_model/ntu100_pretrained.pth'

# 输入测试集的地方，在cfg里
cfg = Config.fromfile('/home/code/video_model/create_metric_data_config.py')
cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test, dict(test_mode=True))
dataloader_setting = dict(
    videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
    workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
    shuffle=False)
dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
data_loader = build_dataloader(dataset, **dataloader_setting)
# test_loader = torch.utils.data.DataLoader(
#         )

from model.builder import build_model
net = build_model(cfg.model)

if model_file:
    print('=> loading model: {}'.format(model_file))
    # net.load_state_dict(torch.load(model_file))
    load_checkpoint(net, model_file, map_location='cuda:0')
    print('=> tesing model...')

model = net.to(device)

data_test_list = []
data_sample_list = []
label_test_list = []
label_sample_list = []


model.eval()
for i in data_loader:
    # data = i['keypoint'][0].cuda()
    # label = i['label'].cuda()
    data = i['keypoint'][0].to(device) 
    label = i['label']
    res = model(keypoint=i['keypoint'].to(device),return_loss=False) # 使用model embed
    data_test_list.append(res)
    label_test_list.append(label)


# 将测试或锚点数据转换为embedding后，保存为pickle文件
# label_20_test_ntu100_pretrained.pkl保存测试数据标签，label_20_sample_ntu100_pretrained.pkl保存锚点数据标签
with open('label_20_test_ntu100_pretrained_new.pkl', 'wb') as f: 
    pickle.dump(label_test_list,f)
# vector_20_test_ntu100_pretrained.pkl保存测试数据，vector_20_sample_ntu100_pretrained.pkl保存锚点数据
with open('vector_20_test_ntu100_pretrained_new.pkl', 'wb') as f:
    pickle.dump(data_test_list,f)
