import numpy as np
from mmcv import load

def test_use_distance():
    sample = load('/home/code/video_model/test_metric/vector_20_sample_ntu100_pretrained.pkl') # sample
    sample_label = load('/home/code/video_model/test_metric/label_20_sample_ntu100_pretrained.pkl')
    test_dataset = load('/home/code/video_model/vector_20_test_ntu100_pretrained_new.pkl') # test_dataset,里面包含所有样本的1024特征向量
    test_label = load('/home/code/video_model/label_20_test_ntu100_pretrained_new.pkl') # test_label,dataset里的每个样本类别和label里一一对应，calculate一下distance，把distance最小的那个类别

    num_classes = 120
    class_vectors_test = [[] for i in range(num_classes)] # 以label_120中的内容为索引,存储vector_120中的内容
    class_vectors_sample = [[] for i in range(num_classes)]
    # 将特征向量添加到相应类别的数组列表中
    for i, feat in enumerate(test_dataset):
        class_vectors_test[test_label[i].item()].append(feat) # 将对应类别的向量分到对应类别的列表下
    for i, feat in enumerate(sample):
        class_vectors_sample[sample_label[i].item()].append(feat)

    center_vectors = []
    hit = 0

    for i,vecs in enumerate(class_vectors_sample):
        if vecs != []:
            tmp = dict()
            tmp['label'] = i
            tmp['data'] = (np.ravel(np.mean(vecs, axis=0)))
            center_vectors.append(tmp)
    # 计算每个类别的样本向量和vector的距离之和
    for gt, vecs in enumerate(class_vectors_test): # vecs包含当前遍历到的类别所有样本vec
        for instance in vecs:
            dis = []
            for center in center_vectors:
                dis.append(np.sqrt(np.sum((instance-center['data'])**2)))
            dis = np.array(dis)
            pred = np.argmin(dis)
            pred = center_vectors[pred]['label']
            if gt == pred:
                hit += 1

    esp = 1e-05
    accuracy = (hit + esp) / len(test_dataset)

    # print(accuracy)
    return accuracy
    

if __name__ == '__main__':

    acc = test_use_distance()
    
    print(f"Accuracy: {acc*100} %")
