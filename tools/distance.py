import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import torch
import mmcv
from mmcv import load
from utils.inference import inference_recognizer, init_recognizer
import argparse


def get_fer_data():
    data = load("/home/shihr/code/video_model/tools/vector_120.pkl")
    label = load('/home/shihr/code/video_model/tools/label_120.pkl')

    vector = load('/home/shihr/code/video_model/vector.pkl')

    num_classes = 60
    class_vectors = [[] for i in range(num_classes)]

    # 将特征向量添加到相应类别的数组列表中
    for i, feat in enumerate(data):
        class_vectors[label[i].item()].append(feat)

    # 计算每个类别的中心向量
    center_vectors = []
    distances=[]
    for vecs in class_vectors:
        center_vectors.append(np.ravel(np.mean(vecs, axis=0)))
    for vecs in class_vectors:
        distances.append(np.sqrt(np.sum((vecs-vector)**2)))
    print(distances)
    # n_samples, n_features = data.shape
    center_vectors.append(np.ravel(vector))
    with open('new_vector.pkl', 'wb') as f:
        pickle.dump(center_vectors,f)
    # return data, label, n_samples, n_features
    return center_vectors, distances

color_map = ['r','y','k','g','b',
             'm','c','brown','darkorange','darkgreen',
             'darkcyan','blueviolet','fuchsia','yellowgreen','turquoise',
             'tan','dimgrey','tomato','coral','pink',
             'dodgerblue','darkturquoise','mediumpurple','violet','silver'] 
def plot_embedding_2D(data, label, title):

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    data = np.array(data) ######
    for i in range(data.shape[0]):
        if i<21:
            plt.plot(data[i, 0], data[i, 1],marker='o',markersize=4,color=color_map[i])
        else:
            plt.plot(data[i, 0], data[i, 1],marker='o',markersize=4,color=color_map[i%24])
        
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig("/home/shihr/code/video_model/vis_cs_120.png")
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description='demo')

    parser.add_argument(
        '--config',
        default='/home/shihr/code/video_model/.vscode/infer/infer_config.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default=('/home/shihr/code/video_model/.vscode/resources/pretrained_model.pth'))
    parser.add_argument(
        '--pose-config',
        default='/home/shihr/code/video_model/.vscode/resources/hrnet_coco.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--device', type=str, default='cuda:1', help='CPU/CUDA device option')

    args = parser.parse_args()
    return args


from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result
def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret


def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))


from scipy.optimize import linear_sum_assignment
def pose_tracking(pose_results, max_tracks=2, thre=30):
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]
        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]['data'][-1][1], poses[j])

        row, col = linear_sum_assignment(scores)
        for r, c in zip(row, col):
            track_proposals[r]['data'].append((idx, poses[c]))
        if m > n:
            for j in range(m):
                if j not in col:
                    num_tracks += 1
                    new_track = dict(data=[])
                    new_track['track_id'] = num_tracks
                    new_track['data'] = [(idx, poses[j])]
                    tracks.append(new_track)
    tracks.sort(key=lambda x: -len(x['data']))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            idx, pose = item
            result[i, idx] = pose
    return result[..., :2], result[..., 2]


def caculate_distance(frame_paths, det_results, h, w ,num_frame):
    args = parse_args()

    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
    GCN_flag = 'GCN' in config.model.type
    GCN_nperson = None
    if GCN_flag:
        format_op = [op for op in config.data.test.pipeline if op['type'] == 'FormatGCNInput'][0]
        GCN_nperson = format_op['num_person']
    model = init_recognizer(config, args.checkpoint, args.device)

    pose_results = pose_inference(args, frame_paths, det_results)
    with torch.cuda.device(args.device):
        torch.cuda.empty_cache()

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

    if GCN_flag:
        # We will keep at most `GCN_nperson` persons per frame.
        tracking_inputs = [[pose['keypoints'] for pose in poses] for poses in pose_results]
        keypoint, keypoint_score = pose_tracking(tracking_inputs, max_tracks=GCN_nperson)
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score
    else:
        num_person = max([len(x) for x in pose_results])
        # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
        num_keypoint = 17
        keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                            dtype=np.float16)
        keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                  dtype=np.float16)
        for i, poses in enumerate(pose_results):
            for j, pose in enumerate(poses):
                pose = pose['keypoints']
                keypoint[j, i] = pose[:, :2]
                keypoint_score[j, i] = pose[:, 2]
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score

    inference_recognizer(model, fake_anno, outputs='cls_head')

    data, label = get_fer_data()
    print('Begining......')

	# 调用t-SNE对高维的data进行降维，得到的2维的result_2D，shape=(samples,2)
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0) 
    result_2D = tsne_2D.fit_transform(data)
    
    print('Finished......')
    fig1 = plot_embedding_2D(result_2D, label, 'feature vector center')	# 将二维数据用plt绘制出来
    fig1.show()
    
    # plt.pause(50)
    

# if __name__ == '__main__':
#     caculate_distance()

