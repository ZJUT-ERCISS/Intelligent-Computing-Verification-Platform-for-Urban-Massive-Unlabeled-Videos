import os
import os.path as osp
import cv2
import decord
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from mmcv import load
from tqdm import tqdm
import copy as cp

from utils.util import mrlines, merge_time, nms, get_final_score, get_proposal_dict
from config.config import cfg


try:
    from mmpose.apis import inference_top_down_pose_model
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')


try:
    from mmtrack.apis import inference_mot, init_model
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmtrack.apis`. These apis are '
                      'required in this script! ')


def mot_infer(model, frames):
    results = []
    for i, frame in enumerate(frames):
        result = inference_mot(model, frame, i)
        results.append(result)
    return results


def extract_frame(video_path): 
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]


def extract_frame_down(video_path): 
    vid = decord.VideoReader(video_path)
    CAP = cv2.VideoCapture(video_path)
    FPS = CAP.get(cv2.CAP_PROP_FPS)
    frames = []
    count = 0
    for x in vid:
        if count % FPS == 0:
            frames.append(x.asnumpy())
        count+=1
    return frames


def detection_inference(model, frames):
    results = []
    for frame in frames:
        result = inference_detector(model, frame) # 使用model单独对单帧进行目标检测的推理
        results.append(result)
    return results


def pose_inference(anno_in, model, frames, det_results):

    anno = cp.deepcopy(anno_in) 
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    num_person = max([len(x) for x in det_results]) 
    anno['total_frames'] = total_frames
    anno['num_person_raw'] = num_person

    kp = np.zeros((num_person, total_frames, 17, 3), dtype=np.float32) # 关键点shape(M, T, V, C)
    for i, (f, d) in enumerate(zip(frames, det_results)): # i index, f frame, d det_result
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        for j, item in enumerate(pose):
            kp[j, i] = item['keypoints']
        anno['keypoint'] = kp[..., :2].astype(np.float16)
        anno['keypoint_score'] = kp[..., 2].astype(np.float16)

    return anno


def draw_gt_and_proposal(save_root: str, file_name: str, keypoint_score: list, fps, proposals: list=[]):
    
    ncol = 4
    # 所有人的节点分数放入一个图内
    fig_gt_prop, ax = plt.subplots()
    # 按人物划分
    for m in range(keypoint_score.shape[0]):
        ax.bar(np.arange(1, keypoint_score.shape[1]+1), [np.mean(s) for _, s in enumerate(keypoint_score[m, :, :])], \
               label='person{}_gt'.format(m+1))

    ax.set_title('gt and proposals')
    ax.set_xlabel('snippet')
    ax.set_ylabel('score')
    ax.set_xlim([0,keypoint_score.shape[1]])
    ax.set_ylim([0,1])
    # ax.legend()
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    
    if not os.path.isdir(save_root+"/{}".format(file_name)):
            os.mkdir(save_root+"/{}".format(file_name))
    # 看proposals
    if proposals:
        for _, time in enumerate(proposals):
        
            # print(time)
            frametoStart = np.round(time[0] * fps)
            frametoEnd = np.round(time[1] * fps)
            frame = np.arange(int(frametoStart), int(frametoEnd+1))
            y_axis = [0.9] * len(frame)
            ax.plot(frame, y_axis, color='red', label = 'proposal')

    fig_gt_prop.savefig(save_root+"/{}".format(file_name)+"/gt_all_and_proposal.png", dpi=500, bbox_inches = 'tight')
    plt.close(fig_gt_prop)

    return 1


def Vis2DPose(item, thre=0.2, out_shape=(540, 960), layout='coco', fps=24, video=None):
    if isinstance(item, str):
        item = load(item)

    assert layout == 'coco'

    kp = item['keypoint']
    if 'keypoint_score' in item:
        kpscore = item['keypoint_score']
        kp = np.concatenate([kp, kpscore[..., None]], -1)

    assert kp.shape[-1] == 3
    img_shape = item.get('img_shape', out_shape)
    kp[..., 0] *= out_shape[1] / img_shape[1]
    kp[..., 1] *= out_shape[0] / img_shape[0]

    total_frames = item.get('total_frames', kp.shape[1])
    assert total_frames == kp.shape[1]

    if video is None:
        frames = [np.ones([out_shape[0], out_shape[1], 3], dtype=np.uint8) * 255 for i in range(total_frames)]
    else:
        vid = decord.VideoReader(video)
        frames = [x.asnumpy() for x in vid]
        frames = [cv2.resize(x, (out_shape[1], out_shape[0])) for x in frames]
        if len(frames) != total_frames:
            frames = [frames[int(i / total_frames * len(frames))] for i in range(total_frames)]

    if layout == 'coco':
        edges = [
            (0, 1, 'f'), (0, 2, 'f'), (1, 3, 'f'), (2, 4, 'f'), (0, 5, 't'), (0, 6, 't'),
            (5, 7, 'ru'), (6, 8, 'lu'), (7, 9, 'ru'), (8, 10, 'lu'), (5, 11, 't'), (6, 12, 't'),
            (11, 13, 'ld'), (12, 14, 'rd'), (13, 15, 'ld'), (14, 16, 'rd')
        ]
    color_map = {
        'ru': ((0, 0x96, 0xc7), (0x3, 0x4, 0x5e)),
        'rd': ((0xca, 0xf0, 0xf8), (0x48, 0xca, 0xe4)),
        'lu': ((0x9d, 0x2, 0x8), (0x3, 0x7, 0x1e)),
        'ld': ((0xff, 0xba, 0x8), (0xe8, 0x5d, 0x4)),
        't': ((0xee, 0x8b, 0x98), (0xd9, 0x4, 0x29)),
        'f': ((0x8d, 0x99, 0xae), (0x2b, 0x2d, 0x42))}

    for i in tqdm(range(total_frames)):
        for m in range(kp.shape[0]):
            ske = kp[m, i]
            for e in edges:
                st, ed, co = e
                co_tup = color_map[co]
                j1, j2 = ske[st], ske[ed]
                j1x, j1y, j2x, j2y = int(j1[0]), int(j1[1]), int(j2[0]), int(j2[1])
                conf = min(j1[2], j2[2])
                if conf > thre:
                    color = [x + (y - x) * (conf - thre) / 0.8 for x, y in zip(co_tup[0], co_tup[1])]
                    color = tuple([int(x) for x in color])
                    frames[i] = cv2.line(frames[i], (j1x, j1y), (j2x, j2y), color, thickness=2)
    return mpy.ImageSequenceClip(frames, fps=fps)


def sample_video_slice(video_path: str, output_dir: str, segments_time: list):
    '''
    两个高质量节点片段之间间隔大于一定的帧数再将其分离
    '''

    ext = os.path.basename(video_path).strip().split('.')[-1]
    assert ext in ['mp4', 'avi', 'flv']
    cap = cv2.VideoCapture(video_path)  # 读取视频文件
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    if ext in ['mp4']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'xvid')
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    current = 0
    output_video = []

    basename_without_ext = os.path.splitext(os.path.basename(video_path))[0]
    
    for i, seg_t in enumerate(segments_time):

        frameToStart = round(seg_t[0] * fps)
        frametoStop = round(seg_t[-1] * fps)
        
        # print(current)
        # print(frametoStop)

        # output_video.append(os.path.join(output_dir, '{}_slice{}.{}'.format(basename_without_ext, i + 1, ext)))
        
        time_str = "(" + str(round(seg_t[0],1)) + "-" + str(round(seg_t[1],1)) + ")"
        output_video.append(os.path.join(output_dir, '{}_slice{}_{}.{}'.format(basename_without_ext, i+1, time_str, ext)))
        
        
        current = frameToStart
        cap.set(cv2.CAP_PROP_POS_FRAMES, current)
        # 创建一个输出视频的写入器
        video_writer =cv2.VideoWriter(output_video[i], fourcc, fps, frame_size)
        while (current >= frameToStart and current < frametoStop):
            success, frame = cap.read()
            if success:
                # print('current_frame= ', current)
                video_writer.write(frame)
                current += 1
            else:
                break
        video_writer.release()
    cap.release()

    return output_video


def locate_and_cut_video(video_list, output_dir, config, save_fig=True):

    lines = mrlines(video_list)
    lines = [x.split() for x in lines]

    # * We set 'frame_dir' as the base name (w/o. suffix) of each video
    assert len(lines[0]) in [1, 2]
    if len(lines[0]) == 1:
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0]) for x in lines] 
    else:
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0], label=int(x[1])) for x in lines]

    my_part = annos
    os.makedirs(config.TEMP_DIR, exist_ok=True)

    det_model = init_detector(config.DET_CONFIG, config.DET_CKPT, 'cuda:1') 
    assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'
    pose_model = init_pose_model(config.POSE_CONFIG, config.POSE_CKPT, 'cuda:1') 

    for anno in tqdm(my_part):
        # frames = extract_frame(anno['filename'])
        frames = extract_frame_down(anno['filename'])
        det_results = detection_inference(det_model, frames) 
        det_results = [x[0] for x in det_results]
        for i, res in enumerate(det_results):
            res = res[res[:, 4] >= config.DET_SCORE_TH]
            box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0]) # bounding box
            assert np.all(box_areas >= 0)
            res = res[box_areas >= config.DET_AREA_TH]
            det_results[i] = res
        shape = frames[0].shape[:2]
        anno['img_shape'] = shape
        anno = pose_inference(anno, pose_model, frames, det_results) 
   
        if not os.path.isdir(output_dir + '/raw'):
            os.mkdir(output_dir + '/raw')
        if not os.path.isdir(output_dir + '/raw' + '/{}'.format(anno['frame_dir'])):
            os.mkdir(output_dir + '/raw' + '/{}'.format(anno['frame_dir']))
        # print(anno)
        video_path = anno['filename']
        anno.pop('filename')
        CAP = cv2.VideoCapture(video_path)
        FPS = CAP.get(cv2.CAP_PROP_FPS)
        FRAMES = CAP.get(cv2.CAP_PROP_FRAME_COUNT)

        final_score = get_final_score(anno['keypoint_score'])
        proposal_dict = get_proposal_dict(anno['keypoint_score'], final_score, FRAMES, FRAMES/FPS, FPS, cfg)
        final_proposals = [nms(v, config.NMS_THRESH) for _, v in proposal_dict.items()]
        # merge_proposals =  merge_time(final_proposals, 0.25, 1.0)
        merge_proposals =  merge_time(final_proposals, 0.25, 0.01)

        process_list = []
        for i, time in enumerate(merge_proposals):
            start = time[0] * FPS
            end = time[1] * FPS
            while end - start > 60.0:
                mark = [start, start+60]
                process_list.append(mark)
                start += 60
            process_list.append([round(start, 2), round(end, 2)])
        merge_proposals = process_list

        if save_fig:
            if not os.path.isdir('./quality_figure'):
                os.mkdir('./quality_figure')
            draw_gt_and_proposal('./quality_figure', anno['frame_dir'], anno['keypoint_score'], FPS, merge_proposals)
        
        sample_video_slice(video_path=video_path, output_dir=output_dir + '/raw' + '/{}'.format(anno['frame_dir']), segments_time=merge_proposals)
    return 1


def vis_sp_skeleton(cut_video_list, output_dir, config, layer_only=True, alpha=0.8):
    '''
    alpha: 是单个人在整个切片中的时间长度占比阈值，即小于alpha倍原视频长度的单人片段都可显示
    '''
    lines = mrlines(cut_video_list)
    lines = [x.split() for x in lines]

    # * We set 'frame_dir' as the base name (w/o. suffix) of each video
    assert len(lines[0]) in [1, 2]
    if len(lines[0]) == 1:
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0]) for x in lines] 
    else:
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0], label=int(x[1])) for x in lines]
    # print(annos)

    my_part = annos
    os.makedirs(config.TEMP_DIR, exist_ok=True)

    # print(my_part)

    mot_model = init_model(config.MOT_CONFIG, config.MOT_CKPT, 'cuda') 
    pose_model = init_pose_model(config.POSE_CONFIG, config.POSE_CKPT, 'cuda') 

    results = []
    for anno in tqdm(my_part): 
        frames = extract_frame(anno['filename'])
        mot_results = mot_infer(mot_model, frames) 

        track_results = []
        for i, res in enumerate(mot_results):
            track_results.append(res['track_bboxes'][0])

        for i, res in enumerate(track_results):
            res = res[res[:, 5] >= config.DET_SCORE_TH]
            box_areas = (res[:, 4] - res[:, 2]) * (res[:, 3] - res[:, 1])
            assert np.all(box_areas >= 0)
            res = res[box_areas >= config.DET_AREA_TH]
            track_results[i] = res
        # 统计切片中出现的总人数
        if len(track_results) != 0:
            anno['num_person_raw'] = int(max(res[:, 0].max() for res in track_results if len(res) != 0) + 1)
            # anno['num_person_raw'] = int(max(res[:, 0].max() for res in track_results) + 1)
        else:
            continue
        
        shape = frames[0].shape[:2]
        anno['img_shape'] = shape
        bboxes_personid = [[] for _ in range(anno['num_person_raw'])]
        # 将track的结果按id整理
        for i, res in enumerate(track_results):
            for _, bbox in enumerate(res):
                bboxes_personid[int(bbox[0])].append(np.array([bbox[1:]], dtype=np.float32))
            for m in range(len(bboxes_personid)):
                if len(bboxes_personid[m]) != i+1:
                    bboxes_personid[m].append(np.array([[0]*5], dtype=np.float32)) 
        # 对每个id的人姿态估计
        for m, person_res in enumerate(bboxes_personid):
            temp = anno
            temp['person_id'] = m
            if cfg.FILTER_P:
                if sum(1 for arr in person_res if np.all(arr==0)) < len(frames) * alpha:
                    temp = pose_inference(temp, pose_model, frames, person_res)
                    results.append(temp)
            else:
                continue

    for anno in results:
        FPS = cv2.VideoCapture(anno['filename']).get(cv2.CAP_PROP_FPS)
        if layer_only:
            if not os.path.isdir(output_dir + '/skeleton/single_person' + '/{}'.format(anno['frame_dir'])):
                os.mkdir(output_dir + '/skeleton/single_person' + '/{}'.format(anno['frame_dir']))
            output_path = output_dir + '/skeleton/single_person' + '/{}/{}_person_No{}.mp4'.format(anno['frame_dir'], anno['frame_dir'], anno['person_id']+1)
            vid = Vis2DPose(anno, thre=0.2, out_shape=anno['img_shape'], layout='coco', fps=FPS, video=None)
        else:
            if not os.path.isdir(output_dir + '/raw_w_layer/single_person' + '/{}'.format(anno['frame_dir'])):
                os.mkdir(output_dir + '/raw_w_layer/single_person' + '/{}'.format(anno['frame_dir']))
            output_path = output_dir + '/raw_w_layer/single_person' + '/{}/{}_person_No{}.mp4'.format(anno['frame_dir'], anno['frame_dir'], anno['person_id']+1)
            vid = Vis2DPose(anno, thre=0.2, out_shape=anno['img_shape'], layout='coco', fps=FPS, video=anno['filename'])
        vid.write_videofile(output_path, codec='libx264', audio=False)

    return 1

