import numpy as np
import cv2
from utils.util import get_final_score, get_proposal_dict, nms, merge_time
from config.config import cfg
from mmcv import dump, load

def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f


def calculate_avg_recall(predictions, gt):
    
    total_ar = 0
    for idx, pred in enumerate(predictions):
        inter = 0
        for p in pred:
            # print(p)
            max = np.maximum(gt[idx][0], p[0])
            min = np.minimum(gt[idx][1], p[1])
            if min - max > 0:
                inter += (min - max)
        pred_ar = float(inter + 1e-5) / (gt[idx][1] - gt[idx][0])
        total_ar += pred_ar
    avg_ar = total_ar / len(gt)
    return avg_ar


def calculate_tiou(prediction, gt):
    '''
    prediction: list, 一个视频的merge_time
    gt: list
    '''
    inter = 0
    range_p = 0
    # 每个人的框与gt框的交集,右边界取最小，左边界取最大
    # print(prediction)
    for p in prediction:
        max = np.maximum(gt[0], p[0])
        min = np.minimum(gt[1], p[1])
        if min - max > 0:
            inter += (min - max)
            # print(inter)
        range_p += (p[1] - p[0])

    union = range_p + (gt[1] - gt[0]) - inter
    
    # print(inter)
    esp = 1e-5
    tiou = float(inter + esp) / union

    # print(tiou)
    return tiou


def calculate_tiou_ap(predictions, gt, tiou_th=0.6):
    
    # print(predictions)
    # print(gt)
    pos_cnt = 0
    for idx, pred in enumerate(predictions):
        if pred == []:
            continue
        tiou = calculate_tiou(pred, gt[idx])
        if tiou >= tiou_th:
            pos_cnt += 1
    # print(pos_cnt)
    tiou_ap = float(pos_cnt + 1e-5) / len(predictions)

    return tiou_ap


def calculate_avg_tiou(predictions, gt):
    
    total_tiou = 0
    for idx, pred in enumerate(predictions):
        if pred == []:
            continue
        tiou = calculate_tiou(pred, gt[idx])
        total_tiou += tiou

    avg_tiou = float(total_tiou + 1e-5) / len(predictions)

    return avg_tiou


if __name__ == '__main__':
    annos = load('/home/code/video_model/test_skloc/video_list_activitynet1-3_176.pkl')
    lines = mrlines('/home/code/video_model/test_skloc/video_list_activitynet1-3_176.list')
    gt_annos = load('/home/code/video_model/test_skloc/annotations_activitynet1-3_176.pkl')
    predictions = list()

    for i, anno in enumerate(annos):
        # video_path = '/home/shy/skeleton_location/data/176/' + anno['frame_dir'] + '.'
        video_path = lines[i]
        # print(video_path)
        CAP = cv2.VideoCapture(video_path)
        FPS = CAP.get(cv2.CAP_PROP_FPS)
        # print(FPS)
        FRAMES = CAP.get(cv2.CAP_PROP_FRAME_COUNT)
        # print(FRAMES)

        global_score = get_final_score(anno['keypoint_score'])
        proposal_dict = get_proposal_dict(anno['keypoint_score'], global_score, FRAMES, FRAMES/FPS, FPS, cfg)
        # print(proposal_dict)
        final_proposals = [nms(v, cfg.NMS_THRESH) for _, v in proposal_dict.items()]
        # print(final_proposals)
        merge_proposals =  merge_time(final_proposals, 0.25, 1.0)
        # print(merge_proposals)
        predictions.append(merge_proposals)


    tiou_th = 0.5
    tiou_ap = calculate_tiou_ap(predictions, gt_annos['segment'], tiou_th)

    # avg_tiou = calculate_avg_tiou(predictions, gt_annos['segment'])
    # avg_ar = calculate_avg_recall(predictions, gt_annos['segment'])

    print(f"The prediction rate for tiou is greater than or equal to {tiou_th}: {tiou_ap}")
    # print(f"AR: {avg_ar}")
    # print(f"average tiou: {avg_tiou}")
    


