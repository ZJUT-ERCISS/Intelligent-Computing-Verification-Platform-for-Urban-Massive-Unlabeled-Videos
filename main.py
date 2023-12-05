import os
import argparse

from config.config import cfg
from utils.util import video2list, mrlines
from cut_video.vis import locate_and_cut_video, vis_sp_skeleton
from engine.infer_action import infer_video


def vis_slice_spskeleton(video_dir, out_dir, config, enable):
    '''
    生成节点分数高的视频切片和单人骨骼视频

    video_dir: 保存多个视频文件的目录
    out_dir: 包含原始切片视频和单人骨骼视频的根目录
    '''
    video_list = video2list(video_dir, './video_list') 
    print(video_list)
    locate_and_cut_video(video_list , out_dir, config)
    video_slice_list = video2list(out_dir + '/raw', './video_slice_list')
    print(video_slice_list)
    if enable:
        vis_sp_skeleton(video_slice_list, out_dir, config)



def parse_args():
    parser = argparse.ArgumentParser(description='demo')

    parser.add_argument('--video_dir', default='/home/code/video_model/tmp')# 存放原始视频文件的目录
    # parser.add_argument('--video_dir', default='/home/tmp/upload')# 存放原始视频文件的目录
    parser.add_argument('--out_dir', default='/home/tmp/cut')# 储存原始切片视频和单人骨骼视频的根目录
    # parser.add_argument('--video_infer', default='/home/shihr/code/video_model/.vscode/script/video_infer.sh')
    parser.add_argument('--vis', default=False)#单人骨骼片段

    args = parser.parse_args()
    return args


def main():
    print("start")
    args = parse_args()

    video_dir = args.video_dir # 存放原始视频文件的目录
    out_dir = args.out_dir # 储存原始切片视频和单人骨骼视频的根目录

    # 删除文件夹中的所有文件
    # for fn in os.listdir(out_dir):
    #     fp = os.path.join(out_dir, fn)
    #     if os.path.isfile(fp):
    #         os.remove(fp)

    enable_vis = args.vis
    vis_slice_spskeleton(video_dir, out_dir, cfg, enable_vis)
    list_path = './video_slice_list'
    
    video_list = './video_list'
    video_lines = mrlines(video_list)
    names = []

    for line in video_lines:
        name = line.split('/')[len(line.split('/'))-1].split('.')[0]
        names.append(name)
    
    lines = mrlines(list_path)
    lines = [x.split() for x in lines]

    process_lines = []

    for line in lines:
        name = line[0].split('/')[len(line[0].split('/'))-1].split('.')[0]
        for x in names:
            if name.find(x)!=-1:
                process_lines.append(line)

    # for line in lines:
    for line in process_lines:
        splits = line[0].split('/')
        file_name = splits[len(splits)-1]
        dir_path = "/home/tmp/res/" + file_name.split('_')[0]
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        res_path = dir_path + "/infer_" + file_name
        # infer_path = args.video_infer
        # cmd = "bash /home/shihr/code/video_model/.vscode/script/video_infer.sh " + line[0] +" "+ res_path
        # cmd = "bash " + infer_path + " " + line[0] +" "+ res_path
        # os.system(command=cmd)
        infer_video(line[0], res_path)

    print("finish")


if __name__ == '__main__':
    main()
    

