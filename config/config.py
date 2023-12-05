# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import numpy as np
import os
from easydict import EasyDict as edict

cfg = edict()

cfg.GPU_ID = '1'

cfg.DET_CONFIG = f'/home/code/video_model/config/mmdet/faster_rcnn_hrnetv2p_w32_2x_coco.py'
cfg.DET_CKPT = f'/home/code/video_model/resources/ckpt/faster_rcnn_hrnetv2p_w32_2x_coco_20200529_015927-976a9c15.pth'

cfg.POSE_CONFIG = f'/home/code/video_model/config/mmpose/hrnet_w32_coco_256x192.py'
cfg.POSE_CKPT = f'/home/code/video_model/resources/ckpt/hrnet_w32_coco_256x192-c78dce93_20200708.pth'

cfg.MOT_CONFIG = f'/home/code/video_model/config/mmtrack/mot/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py'
cfg.MOT_CKPT = '/home/code/video_model/resources/ckpt/tracktor_reid_r50_iter25245-a452f51f.pth'

# cfg.VIDEO_PATH = "/home/shy/skeleton_location/resource/video"
# cfg.VIDEO_LIST = "/home/shy/skeleton_location/video_list"

# cfg.CUT_VIDEO_DIR = "/home/shy/skeleton_location/resource/cut_video"

# cfg.FIG_SAVE_ROOT = "/home/shy/skeleton_location/cas_picture"
cfg.DET_SCORE_TH = 0.7
cfg.DET_AREA_TH = 1600.0
# cfg.OUT_ANNOS_PATH = ''
cfg.LOCAL_RANK = 0
cfg.TEMP_DIR = 'tmp'
cfg.NON_DIST = True
# cfg.COMPRESS = True
# cfg.FEATS_DIM = 2048
# cfg.NUM_WORKERS = 8
cfg.NMS_THRESH = 0.7
cfg.CAS_THRESH = np.arange(0.3, 0.5, 0.02)

cfg.FILTER_P = True
cfg.COMPRESS =False