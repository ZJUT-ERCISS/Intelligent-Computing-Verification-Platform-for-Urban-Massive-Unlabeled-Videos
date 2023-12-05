from mmcv import load, dump
import os
import random

def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f

# ntu_kin = load('/home/shihr/code/ntu_kin.pkl')
# tmp = list()
# ntu120_test_list = ['A001','A007','A013','A019','A025','A031','A037','A043','A049','A055','A061','A067','A073','A079','A085','A091','A097','A103','A109','A115']

# # one shot
# ntu120_sample_list = ['S001C003P008R001A001','S001C003P008R001A007','S001C003P008R001A013','S001C003P008R001A019','S001C003P008R001A025','S001C003P008R001A031',
#                       'S001C003P008R001A037','S001C003P008R001A043','S001C003P008R001A049','S001C003P008R001A055','S018C003P008R001A061','S018C003P008R001A067',
#                       'S018C003P008R001A073','S018C003P008R001A079','S018C003P008R001A085','S018C003P008R001A091','S018C003P008R001A097','S018C003P008R001A103',
#                       'S018C003P008R001A109','S018C003P008R001A115'
#                      ]

arr_train = []
arr_val = []
#arr_sample = []
# 新添加的数据集list
video_list = '/home/code/video_model/video_list2'
lines = mrlines(video_list)
 
# # one-shot 20 classes
# for anno in ntu_kin['annotations']:
#     for val_name in ntu120_test_list:
#         if val_name in anno['frame_dir']:
#             arr_val.append(anno['frame_dir'])
#             tmp.append(anno)
#     for train_name in ntu120_sample_list:
#         if train_name == anno['frame_dir']:
#             arr_train.append(anno['frame_dir'])
#             arr_val.remove(anno['frame_dir'])

# 拼接两个pickle文件
origin_path = '/home/code/video_model/ntu20_8trains_restval_addMCFD.pkl'
# 新添加的数据集
new_data_path = '/home/code/video_model/video_list.pkl'
anno = load(origin_path)
split = anno['split']
#print(anno)
new_anno = load(new_data_path)
#print(new_anno)
anno = anno['annotations'] + new_anno

# 划分训练与测试数据集
for line in lines:
   # print(line.rsplit('/',4)[3])
   # print(line.rsplit('/',4)[4].split()[0].split('.')[0])
    if line.rsplit('/', 4)[3]=='val':
        arr_val.append(line.rsplit('/',4)[4].split()[0].split('.')[0])
    if line.rsplit('/', 4)[3]=='train':
        arr_train.append(line.rsplit('/',4)[4].split()[0].split('.')[0])



#split = dict()
#print(split)
#print(arr_val)
split['xsub_train'] = split['xsub_train'] + arr_train
split['xsub_val'] = split['xsub_val'] + arr_val
#print(split)
#split['xsub_sample'] = arr_sample
# split['xsub_val'] = arr_train

result_path = '/home/code/video_model/new_results.pkl'
dump(dict(split=split, annotations=anno), result_path)
print("finish")

