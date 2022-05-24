from torch import nn


class MetaParam(object):
    def __init__(self):
        self.inner_lr = 0.05                           # 内层学习率
        self.eeg_features_chose_id = 10                # 将33个EEG通道随机排列，取其中一个排列
        self.eeg_top_n = 5                             # 取前n个特征
        self.other_features_chose_id = 10              # 将8个其余通道随机排列，取其中一个排列
        self.other_top_n = 3                           # 取前n个特征


class MlpParam(object):
    def __init__(self):
        self.input_dim = None
        self.hidden_dim = None
        self.output_dim = None
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None

    def init_param(self):
        if self.input_dim and self.hidden_dim and self.output_dim:
            self.w1 = nn.Parameter()
            self.b1 = nn.Parameter()
            self.w2 = nn.Parameter()
            self.b2 = nn.Parameter()
        else:
            raise "didn't have input_dim, hidden_dim or output_dim"

