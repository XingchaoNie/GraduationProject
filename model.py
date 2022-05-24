import torch
import torch.nn as nn
import torch.nn.functional as f


class MetaLinear(nn.Module):
    def __init__(self, init_layer = None):
        super(MetaLinear, self).__init__()
        if type(init_layer) != type(None):
            self.weight = init_layer.weight.clone()
            self.bias = init_layer.bias.clone()
    def zero_grad(self, set_to_none: bool = False) -> None:
        self.weight.grad = torch.zeros_like(self.weight)
        self.bias.grad = torch.zeros_like(self.bias)
    def forward(self,x):
        return f.linear(x, self.weight, self.bias)


class MAML(object):
    # 先用全连接层实现一遍
    def __init__(self, init_weight=None, input_dim=0, n_hidden_1=0, output_dim=0):
        super(MAML, self).__init__()
        self.model = MLP(init_weight, input_dim, n_hidden_1, output_dim)
        if type(init_weight) != type(None):
            self.model.load_state_dict(init_weight)
        self.grad_buffer = 0

    # 为本轮的所有任务建立子模型并返回进行后续操作,且这波子模型的参数全部继承自MAML
    def get_these_models(self, num):
        return [MLP(init_weight=self.model) for i in range(num)]

    def clear_grad_buffer(self):
        print("清空前buffer:", self.grad_buffer)
        self.grad_buffer = 0
        print("已清零")

class MetaLinear(nn.Module):
    def __init__(self, init_layer = None):
        super(MetaLinear, self).__init__()
        if type(init_layer) != type(None):
            self.weight = init_layer.weight.clone()
            self.bias = init_layer.bias.clone()
    def zero_grad(self, set_to_none: bool = False) -> None:
        self.weight.grad = torch.zeros_like(self.weight) #zeros_like生成和目标向量维度相同的0向量
        self.bias.grad = torch.zeros_like(self.bias)
    def forward(self,x):
        return f.linear(x, self.weight, self.bias)

class MLP(nn.Module):
    # 构建一个多层感知机，即全连接层作为基本模型，使用元学习来更新参数
    # 使用批标准化加快收敛速度，使用sigmoid激活函数
    def __init__(self, init_weight=None, input_dim=0, n_hidden_1=0, output_dim=0):
        super(MLP, self).__init__()
        if type(init_weight) != type(None):
            for name, module in init_weight.named_modules():
                if name != '':
                    setattr(self, name, MetaLinear(module))

        else:
            self.layer1 = nn.Linear(input_dim, n_hidden_1)
            self.layer2 = nn.Linear(n_hidden_1, output_dim)

    def zero_grad(self, set_to_none: bool = False) -> None:
        layers = self.__dict__["_modules"]
        # 重写为每一层梯度归零
        for layer_key in layers.keys():
            layers[layer_key].zero_grad()

    def update(self, parent, lr):
        layers = self.__dict__['_modules']
        parent_layers = parent.__dict__['_modules']
        for layer_key in layers.keys():
            layers[layer_key].weight = layers[layer_key].weight - lr * parent_layers[layer_key].weight.grad
            layers[layer_key].bias = layers[layer_key].bias - lr * parent_layers[layer_key].bias.grad


    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = self.layer2(x)
        return x


