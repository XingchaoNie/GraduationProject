import os

import numpy as np
import _pickle as cPickle
import pickle
# 这个函数很奇怪啊，我明明安装了都可以用了但是PyCharm给我画红线
from scipy.stats import differential_entropy as de
from utils import high_low_trans

from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd


class Tasks(object):
    def __init__(self):
        self.raw_tasks = list()
        self.train_loader_V = list()
        self.test_loader_V = list()

        self.train_loader_A = list()
        self.test_loader_A = list()

        self.train_loader_D = list()
        self.test_loader_D = list()

        self.train_loader_L = list()
        self.test_loader_L = list()

        self.loader = None
        self.train_loader = None
        self.test_loader = None


    def tasks_preprocessing(self):
        # 拆分标签，将valence, arousal, dominance, like or not的1-9连续评分均拆分成高1低0, 1-5为低， 5-9为高，以此为基准进行任务拆分
        # 核心代码在服务器环境上跑，数据预处理代码在本地环境跑，数据集比较大，不上传到服务器端，只在本地用，所以数据集路径比较乱
        data_path = r'E:\Python_space\SEED-V-multimodal\data_preprocessed_python\data_preprocessed_python'
        self.raw_tasks = {"data": list(), "V_label": list(), "A_label": list(), "D_label": list(), "L_label": list()}
        for i in range(32):
            r = r'\s' + str(i + 1).zfill(2) + '.dat'
            print(r)
            with open(data_path + r, "rb") as f:
                try:
                    this_file = cPickle.load(f, encoding="bytes")
                except cPickle.UnpicklingError:
                    print(r + "is a special file, pass")

            print("preprocessing task:", i + 1)
            this_data = this_file[b'data']
            this_labels = high_low_trans(this_file[b'labels'])
            self.raw_tasks["V_label"].append(this_labels[:, 0:1].copy())
            self.raw_tasks["A_label"].append(this_labels[:, 1:2].copy())
            self.raw_tasks["D_label"].append(this_labels[:, 2:3].copy())
            self.raw_tasks["L_label"].append(this_labels[:, 3:4].copy())
            # 特征提取,缩减特征长度
            split_len = 1  # 每1秒提取一次特征
            this_features = list()
            this_features.clear()
            # 这里考虑一下各通道特征的贡献度是否均等，根据文献要给不同的特征通道赋予不同的权重
            for j in range(np.shape(this_data)[0]):
                this_tensor = this_data[j]
                # 数据被缩减采样到了128HZ，即每秒包含128行数据，对这些数据每1秒进行一次差分熵特征提取，最终会得到63行的特征
                # de_features = list()
                # de_features.clear()
                features = np.ones((40, int(np.shape(this_tensor)[1] / 128 / split_len)))
                # print(np.shape(this_tensor)[1] / 128 / split_len)
                for m in range(int(np.shape(this_tensor)[1] / 128 / split_len)):
                    # 前32个通道是EEG信号，采用微分熵特征提取
                    features[:32, m] = de(this_tensor[:32, m * 128 * split_len: (m + 1) * 128 * split_len], axis=1)
                    # de_features.append(this_de)
                    # 33，34通道是横向和纵向眼电，采用微分熵特征提取
                    features[32:34, m] = de(this_tensor[32:34, m * 128 * split_len: (m + 1) * 128 * split_len], axis=1)
                    # 35, 36通道是颧骨肌电和斜方肌电，采用微分熵特征提取
                    features[34:36, m] = de(this_tensor[34:36, m * 128 * split_len: (m + 1) * 128 * split_len], axis=1)
                    # 37通道是皮肤电，采用微分熵特征提取出现inf值，换用均值
                    features[36:37, m] = np.var(this_tensor[36:37, m * 128 * split_len: (m + 1) * 128 * split_len],
                                                axis=1)
                    # 38通道是呼吸率，取均值特征提取
                    features[37:38, m] = np.var(this_tensor[37:38, m * 128 * split_len: (m + 1) * 128 * split_len],
                                                axis=1)
                    # 39通道是体积扫描仪数据，取均值特征提取
                    features[38:39, m] = np.var(this_tensor[38:39, m * 128 * split_len: (m + 1) * 128 * split_len],
                                                axis=1)
                    # 40通道是体温数据，取均值特征提取
                    features[39:40, m] = np.var(this_tensor[39:40, m * 128 * split_len: (m + 1) * 128 * split_len],
                                                axis=1)
                this_features.append(features)
            self.raw_tasks["data"].append(np.array(this_features).copy())
            # print("dim 1 is different video/person, dim 2 is different time, dim 3 is different features")
            # print("this_task_shape:", np.shape(raw_tasks[0]))
            print("task", i + 1, "got")
        print("data preprocessed, got ", len(self.raw_tasks["data"]), " tasks")
        with open(r"data\raw_tasks_v3.pkl", 'wb') as f:
            pickle.dump(self.raw_tasks, f)

    def load_raw_tasks(self):
        # 本地端预处理较大的数据，处理完后在服务器端直接用
        with open(r"data/raw_tasks_v1.pkl", 'rb') as f:
            self.raw_tasks = pickle.load(f)
        print(type(self.raw_tasks))
        print(self.raw_tasks.keys())

    def split_and_yield_tasks(self):
        split_point = 1                     # 决定每位受试者数据被等分成为多少个任务
        split = int(40 / split_point)
        raw_tasks_num = len(self.raw_tasks["data"])
        for x in range(raw_tasks_num):
            this_task_data = torch.tensor(self.raw_tasks["data"][x])
            # print(this_task_data.shape)

            this_task_data = torch.transpose(this_task_data, 1, 2)
            # print(this_task_data.shape)

            this_task_V_label = torch.tensor(self.raw_tasks["V_label"][x]).reshape(-1)
            this_task_A_label = torch.tensor(self.raw_tasks["A_label"][x]).reshape(-1)
            this_task_D_label = torch.tensor(self.raw_tasks["D_label"][x]).reshape(-1)
            this_task_L_label = torch.tensor(self.raw_tasks["L_label"][x]).reshape(-1)

            # 小样本任务，所有样本一次性进入，每个batch的大小就是样本总量，进入以后，一半拿来做支持数据，一半拿来做查询数据
            for m in range(split_point):
                this_task_tensor = TensorDataset(this_task_data[split * m: split * (m + 1)], this_task_V_label[split * m: split * (m + 1)])
                self.train_loader_V.append(DataLoader(dataset=this_task_tensor, batch_size=split, shuffle=True))

            for m in range(split_point):
                this_task_tensor = TensorDataset(this_task_data[split * m: split * (m + 1)], this_task_A_label[split * m: split * (m + 1)])
                self.train_loader_A.append(DataLoader(dataset=this_task_tensor, batch_size=split, shuffle=True))

            for m in range(split_point):
                this_task_tensor = TensorDataset(this_task_data[split * m: split * (m + 1)], this_task_D_label[split * m: split * (m + 1)])
                self.train_loader_D.append(DataLoader(dataset=this_task_tensor, batch_size=split, shuffle=True))

            for m in range(split_point):
                this_task_tensor = TensorDataset(this_task_data[split * m: split * (m + 1)], this_task_L_label[split * m: split * (m + 1)])
                self.train_loader_L.append(DataLoader(dataset=this_task_tensor, batch_size=split, shuffle=True))

    def init_loader(self, tag):
        if type(self.loader) != type(None):
            print("loader rebuilt, rejected")
        else:
            if tag == "V":
                self.loader = self.train_loader_V
            elif tag == "A":
                self.loader = self.train_loader_A
            elif tag == "D":
                self.loader = self.train_loader_D
            elif tag == "L":
                self.loader = self.train_loader_L
            else:
                print("error tag")
            print("init_loader done!")

class Tasks2(object):
    def __init__(self):
        super(Tasks2, self).__init__()
        self.top_features_data = None
        self.v_data = None
        self.v_label = None
        self.a_data = None
        self.a_label = None
        self.d_data = None
        self.d_label = None
        self.l_data = None
        self.l_label = None
        self.tasks = None

        self.loaders = None
        self.train_loader = None
        self.test_loader = None

        self.read_data()

    def read_data(self):
        raw_top_v = pd.read_csv('data/valence_data.csv', index_col=0)
        raw_top_a = pd.read_csv('data/arousal_data.csv', index_col=0)
        raw_top_d = pd.read_csv('data/dominance_data.csv', index_col=0)
        raw_top_l = pd.read_csv('data/liking_data.csv', index_col=0)
        raw_label = pd.read_csv('data/labels.csv', index_col=0)
        labels = raw_label[['Valence', 'Arousal', 'Dominance', 'Liking']]
        labels[labels < 4.5] = 0
        labels[labels >= 4.5] = 1
        self.v_data = torch.tensor(np.array(raw_top_v))
        self.a_data = torch.tensor(np.array(raw_top_a))
        self.d_data = torch.tensor(np.array(raw_top_d))
        self.l_data = torch.tensor(np.array(raw_top_l))
        self.v_label = torch.tensor(np.array(labels['Valence']))
        self.a_label = torch.tensor(np.array(labels['Arousal']))
        self.d_label = torch.tensor(np.array(labels['Dominance']))
        self.l_label = torch.tensor(np.array(labels['Liking']))

    def get_tasks_loader(self, tag):
        train_loaders = list()
        test_loaders = list()
        loaders = list()
        if tag == "V":
            data = self.v_data
            label = self.v_label
        elif tag == "A":
            data = self.a_data
            label = self.a_label
        elif tag == "D":
            data = self.d_data
            label = self.d_label
        elif tag == "L":
            data = self.l_data
            label = self.l_label
        else:
            raise "错误的情绪维度" + tag
        for i in range(32):
            # 每位受试者有40条数据
            this_data = data[i*40:(i+1)*40, :]
            this_label = label[i*40:(i+1)*40]
            train_tensor_set = TensorDataset(this_data, this_label)
            loaders.append(DataLoader(dataset=train_tensor_set, batch_size=40, shuffle=True))
        self.loaders = loaders
        print(len(loaders))









if __name__ == '__main__':
    # 测试代码用
    test_tasks = Tasks2()
    # test_tasks.tasks_preprocessing()
    # input()
    test_tasks.get_tasks_loader("V")



