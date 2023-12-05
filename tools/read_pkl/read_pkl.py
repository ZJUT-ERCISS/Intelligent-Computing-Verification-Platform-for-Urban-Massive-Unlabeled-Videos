import mmcv
from mmcv import load
import numpy as np
import pickle

# 
data = mmcv.load('/home/shy/code_hik/video_model/ntu20+fall_test_1sample.pkl')
# with open('/home/shy/code_hik/video_model/.vscode/draw/ntu20.pkl', 'rb') as pickle_file:
#     data = pickle.load(pickle_file)
# print(data[-1].shape)
# print(data[0].shape) # 1,1024
# print(np.mean(data[0], axis=0).shape) # 1024
# print(np.ravel(np.mean(data[0], axis=0)).shape) # 1024
data_str = str(data)
txt_file_path = '/home/shy/code_hik/video_model/tools/read_pkl/ntu20+fall_test_1sample.txt'
with open(txt_file_path, 'w') as txt_file:
    txt_file.write(data_str)