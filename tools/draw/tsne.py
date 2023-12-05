import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
from mmcv import load

def get_fer_data():
    """
	该函数读取上一步保存的两个npy文件，返回data和label数据
    Args:
        data_path:
        label_path:

    Returns:
        data: 样本特征数据，shape=(BS,embed)
        label: 样本标签数据，shape=(BS,)
        n_samples :样本个数
        n_features：样本的特征维度

    """
    # # data = np.load(data_path)
    # data = load('/home/shr/code/gcn_vivit/.vscode/infer/result.pkl')
    # data1 = load('/home/shr/code/gcn_vivit/.vscode/infer/vector.pkl')
    # # label = np.load(label_path)

    # label = []
    # for i in data:
    #     label.append(np.argmax(i))
    # data = load("/home/shr/code/gcn_vivit/.vscode/infer/result.pkl")
    data = load("/home/shr/code/gcn_vivit/.vscode/draw/vector_120.pkl")
    label = load('/home/shr/code/gcn_vivit/.vscode/draw/label_120.pkl')
    # vector = load('/home/shr/code/gcn_vivit/.vscode/infer/vector_point.pkl')
    vector = load('/home/shr/code/gcn_vivit/.vscode/infer/vector.pkl')
    # vector=vector['backbone.to_latent'].to('cpu')
    # vector=vector.numpy()
    # data.append(vector)
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
        
    # plt.plot([data[np.argmin(label), 0], data[21, 0]],[data[np.argmin(label), 1], data[21, 1]])    
        # if label[i]>=120:
        #     index = label[i]-120
        # else:
        #     index = 24
            
        # plt.plot(data[i, 0], data[i, 1],marker='o',markersize=1,color=color_map[index])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig("/home/shr/code/gcn_vivit/.vscode/draw/vis_cs_120.png")
    return fig


def main():
    # data, label, n_samples, n_features = get_fer_data()  # 根据自己的路径合理更改
    data, label = get_fer_data()  # 根据自己的路径合理更改
    print('Begining......') 	

	# 调用t-SNE对高维的data进行降维，得到的2维的result_2D，shape=(samples,2)
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0) 
    # arr = np.empty((2,), dtype=object)
    # arr[0]=np.array(22)
    # arr[1]=np.array(data)
    result_2D = tsne_2D.fit_transform(data)
    
    print('Finished......')
    fig1 = plot_embedding_2D(result_2D, label, 'feature vector center')	# 将二维数据用plt绘制出来
    fig1.show()
    
    # plt.pause(50)
    
if __name__ == '__main__':
    main()

