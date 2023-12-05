import os
import decord
import numpy as np


def video2list(directory, output_file):
    open(output_file, 'w').close()

    with open(output_file, 'w') as file:
        for root, dirs, files in os.walk(directory):
            for file_name in files:
                # 检查文件扩展名是否为视频格式（可以根据需要进行修改）
                if file_name.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    file_path = os.path.join(root, file_name)
                    file.write(file_path + '\n')
    return output_file


def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f


def extract_frame(video_path): 
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]


def nms(proposals, thresh):
    '''
    用IoU筛选片段
    '''
    proposals = np.array(proposals)
    x1 = proposals[:, 2] # proposals按开始时间排序
    x2 = proposals[:, 3]
    scores = proposals[:, 1]

    areas = x2 - x1 + 1
    order = scores.argsort()[::-1] # 对分数从大到小排序，返回分数的索引（argsort为从小到大排序）

    keep = []
    while order.size > 0:
        i = order[0] # 分数最高的片段索引
        keep.append(proposals[i].tolist()) # 添加分数最高的时间段
        # 分数最高的片段与另一个片段（按分数有序排列）的重叠边界xx1, xx2（生成列表）
        xx1 = np.maximum(x1[i], x1[order[1:]]) # intersection左边界
        xx2 = np.minimum(x2[i], x2[order[1:]]) # intersection右边界

        inter = np.maximum(0.0, xx2 - xx1 + 1) # intersection

        # 重叠度
        iou = inter / (areas[i] + areas[order[1:]] - inter) # 根据片段分数由大到小的顺序计算，因为一开始在获得建议框的时候，提供了多个为节点分数服务的阈值
                                                            # 该处应该是计算在不同阈值下得出的片段的重叠度

        # 满足重叠区域小于thresh的片段索引，以该片段为最高分片段重新使用上述方法获得重叠区域
        inds = np.where(iou < thresh)[0] # 找到与当前片段相交程度小的区域，inds为一个包含索引的列表
        order = order[inds + 1] # 选择下一组片段 

    return keep


def check_overlap(intervals: list):
    '''
    判断两个片段的时间区间是否重叠
    '''
    intervals.sort(key=lambda x: x[0])  # 按区间的起始值进行排序
    for i in range(len(intervals) - 1):
        if intervals[i][1] >= intervals[i + 1][0]:
            return True
    return False


def get_final_score(keypoint_score):
    '''
    获得视频级的具体人的得分均值（整个视频）
    '''
    final_score = []
    for m in range(keypoint_score.shape[0]):

        final_score.append(np.mean(keypoint_score[m, :, :]))

    return final_score


def merge_time(final_proposal: list, merge_th: float = -1, duration_th: float=0):
# 融合final_proposal中的时间，只要时间节点
    proposal_time = []

    for _, m in enumerate(final_proposal):
        for i in range(len(m)):
            proposal_time.append([m[i][2], m[i][3]])
        # print(proposal_time)

    proposal_time.sort(key=lambda x: x[0])  # 按区间的起始值进行排序
    if check_overlap(proposal_time):
        merged_time = []
        for t_interval in proposal_time:
            if not merged_time or t_interval[0] > merged_time[-1][1]:
                merged_time.append(t_interval)
            else:
                merged_time[-1] = [merged_time[-1][0], max(merged_time[-1][1], t_interval[1])]
    else:
        merged_time = proposal_time
    
    if merge_th: # 若两个片段之间的差值相差merge_th秒，则合并
        temp = []
        for t_interval in merged_time:
            if not temp or t_interval[0] > temp[-1][1] + merge_th:
                temp.append(t_interval)
            else:
                temp[-1] = [temp[-1][0], max(temp[-1][1], t_interval[1])]
        merged_time = temp

    # 如果区间长度小于duration_th,删除该区间
    if duration_th:
        for _, t_interval in enumerate(merged_time):
            if t_interval[1] - t_interval[0] < duration_th:
                merged_time.remove(t_interval)

    return merged_time


def grouping(arr):
    
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1) 


def get_proposal_oic(tList, wtcam, final_score, scale, v_len, sampling_frames, _lambda=0.25, gamma=0.2):
    
    temp = []
    for m in range(len(tList)): 
        # 确定好人物编号
        m_temp = []
        temp_list = np.array(tList[m])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                if len(grouped_temp_list[j]) < 2:
                    continue  
                t_factor = (16 * v_len) / (scale * len(grouped_temp_list[j]) * sampling_frames) # 16 * 总帧数 / （总时长 * 每个片段包含的帧数 * 帧率）单位是1/帧      
                inner_score = np.mean(wtcam[m, grouped_temp_list[j], :]) # 片段级内部得分
                len_proposal = len(grouped_temp_list[j]) # 长度
                outer_s = max(0, int(grouped_temp_list[j][0] - _lambda * len_proposal)) # max(0, 当前片段的第一帧-lambda*长度)，确定边界
                outer_e = min(int(wtcam.shape[1] - 1), int(grouped_temp_list[j][-1] + _lambda * len_proposal)) # min(wtcam所有帧数-1，当前片段的最后一帧+lambda*长度)，确定边界
                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(range(int(grouped_temp_list[j][-1] + 1), outer_e + 1)) # 外部列表
                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[m, outer_temp_list, 0]) # 外部得分
                m_score = inner_score - outer_score + gamma * final_score[m]
                # t_start = grouped_temp_list[j][0] * t_factor
                # t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                t_start = grouped_temp_list[j][0] / sampling_frames
                t_end = (grouped_temp_list[j][-1] + 1) / sampling_frames
                m_temp.append([m + 1, m_score, t_start, t_end])
            if m_temp == []:
                m_temp.append([m+1, 0, 0, 0])
            temp.append(m_temp)
        # else:
        #     temp.append([[m+1, 0, 0, 0]])
    return temp


def get_proposal_dict(keypoint_score, score_np, vid_num_seg, scale, fps, config):
    '''
    keypoint_score: keypoint_score
    pred: 
    score_np: 视频级得分final_score
    vid_num_seg: 视频的片段数
    '''
    prop_dict = {}
    for th in config.CAS_THRESH:
        kp_tmp = keypoint_score.copy()
        # num_segments = keypoint_score.shape[1]//config.UP_SCALE # 片段包含的帧数，给定的
        # 取片段用的操作，不影响
        for m in range(kp_tmp.shape[0]):
            for f in range(kp_tmp.shape[1]):
                if np.mean(kp_tmp[m, f, :]) < th:
                    kp_tmp[m, f, :] = 0
        seg_list = [np.where(kp_tmp[m, :, 0] > 0) for m in range(kp_tmp.shape[0])] # 表示一个列表，其中每个元素对应一个类别，在该类别下的显著性区域的位置（是列表）表示片段位置。按人分
        proposals = get_proposal_oic(seg_list, keypoint_score, score_np, scale, \
                        vid_num_seg, fps)
        # print(seg_list)
        # print(proposals)
        for i in range(len(proposals)): # len(proposals)为人数
            # print(proposals[i][0][0])
            person_id = proposals[i][0][0]
            prop_dict[person_id] = prop_dict.get(person_id, []) + proposals[i] # 根据不同的thresh生成不同的建议框

    return prop_dict

