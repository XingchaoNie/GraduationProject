"""
author: niexingchao
environment:linux-python3.10
encoding:utf-8
dataset:DEAP
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from random import shuffle, sample, seed
import matplotlib.pyplot as plt

from LoadData import Tasks, Tasks2
from model import MAML, MLP
from config import MlpParam, MetaParam
from utils import label_trans, res_trans
import torch.nn.functional as f


def main():# 后面还有一个top_main函数,这个主函数是第一个版本的，后面有更新
    support = "D"
    model_version = "1"

    # 小任务超参数
    inner_lr = 0.001                         # 每个任务的学习率，MAML产生的模型最终只有一次迭代机会，学习率先放大一点
    inner_epoch_num = 1

    # 元学习参数定义
    tasks_batch_size = 5                                    # 每轮训练抽取任务数量
    outer_epoch_num = 50                                   # 任务迭代次数

    maml_loss_list = list()

    if model_version == "1":
        # 数据预处理
        tasks = Tasks()
        tasks.load_raw_tasks()
        tasks.split_and_yield_tasks()
        tasks.init_loader(support)
        train_tasks_num = int(len(tasks.loader) * 0.85)         # 训练的任务总数
        while train_tasks_num % 5 != 0:
            train_tasks_num = train_tasks_num + 1
        print("have tasks ", len(tasks.loader), "train tasks ", train_tasks_num, "test tasks ", len(tasks.loader) - train_tasks_num)
        shuffle(tasks.loader)
        tasks.train_support_loader = tasks.loader[:train_tasks_num]
        tasks.test_query_loader = tasks.loader[train_tasks_num:]




    # 元学习模型定义,这里直接代表的就是F的输出f
    # MAML的网络是一样的网络，参数什么的都一样，只是个傀儡，这个网络不进行向量运算，损失函数也不一样，梯度方向不是自己的，是一批小任务共同给提供的
    # 这个傀儡每轮要把自己更新的参数给到下一批任务那里，直接赋值即可，网络结构是同一个结构，只是损失函数不一样
    # 好现在架构确定了，我的模型搭建方法不太一样，要自己写，加油！
    # 所有的网络结构参数都在这里调
    maml = MAML(init_weight=None, input_dim=40, n_hidden_1=128, output_dim=2)
    maml_lr = 0.001
    maml_optimizer = torch.optim.Adam(maml.model.parameters(), lr=maml_lr)

    for outer_epoch in range(outer_epoch_num):
        # 进入训练模式
        maml.model.train()
        print("*" * 50 +"epoch", outer_epoch + 1, "..." + "*" * 50)
        # 将任务集随机打乱
        shuffle(tasks.train_support_loader)
        all_these_tasks = list()
        all_these_tasks.clear()
        # 这里是在这个epoch里面把任务的batch分好
        for i in range(int(len(tasks.train_support_loader) / tasks_batch_size)):
            all_these_tasks.append(sample(tasks.train_support_loader, tasks_batch_size))
        # 遍历本次epoch里面的每一个batch
        for these_tasks in all_these_tasks:
            # 使用MAML输出与本轮batch包含任务数目相同的子模型
            these_mlp = maml.get_these_models(len(these_tasks))
            # 这里累加每一个子任务的query损失
            maml_loss = 0
            for task_id, this_task in enumerate(these_tasks):
                # 这里取到了一个任务
                print("training task", task_id + 1, "...")
                these_mlp[task_id].train()
                # MAML模型要求每个模型每次任务只support一次，query一次，将query损失累积起来作为最终损失
                for x, (data, label) in enumerate(this_task):
                    # 这里batch抽取出来，用什么模型对接就可以了，全连接层或者卷积神经网络吧，接着再往下写，用均值损失进行元更新
                    if x > 0:
                        raise "小样本任务batch出问题了"
                    data_num = data.shape[0]
                    support_query_split_point = int(data_num / 2)          # 支持数据与查询数据的分割点
                    support_data = data[:support_query_split_point].reshape((-1, 40))
                    query_data = data[support_query_split_point:].reshape((-1, 40))
                    support_label = label[:support_query_split_point]
                    query_label = label[support_query_split_point:]


                    y = these_mlp[task_id](support_data.to(torch.float32))
                    loss_f = nn.CrossEntropyLoss()
                    loss = loss_f(y.reshape((support_query_split_point, -1)), label_trans(support_label, 2))
                    loss.backward(create_graph=True)
                    these_mlp[task_id].update(lr=1e-2, parent=maml.model)
                    maml_optimizer.zero_grad()

                    y = these_mlp[task_id](query_data.to(torch.float32))
                    loss = loss_f(y.reshape((support_query_split_point, -1)), label_trans(query_label, 2))
                    # 在query上计算的损失不累计到各自的模型里，累加计入maml的损失当中
                    maml_loss = maml_loss + loss

            # 一批任务训练完毕，进行元更新
            maml_loss = maml_loss / tasks_batch_size
            print("maml_loss", maml_loss.item())
            maml_loss.backward(retain_graph=True)
            maml_optimizer.step()
            maml_optimizer.zero_grad()

    # 训练结束，进入测试环节，这里先测试吧，不随便保存模型，测试结果不错之后再保存一次
    print("*" * 100 + "test" + "*" * 100)
    query_num = list()           # 记录每个任务query data总数
    suc_num = list()             # 记录每个任务预测成功的数量
    suc_percent = list()         # 记录每个测试任务的正确率

    for i, test_loader in enumerate(tasks.train_query_loader):
        # 测试的时候挨个抽任务进行测试，由于我这里的主要任务还是利用元学习来进行情绪分类，要在这里测试元学习可以最终提供的准确率
        this_mlp = maml.get_these_models(1)[0]         # 生成1个就行，挨个测试，这里只测试小模型的精度，不测试maml的优度
        for x, (data, label) in enumerate(test_loader):
            if x > 0:
                print("小样本batch出问题了")
            data_num = data.shape[0]
            print(data.shape)
            support_query_split_point = int(data_num * 0.7)  # 支持数据与查询数据的分割点
            support_data = data[:support_query_split_point]
            query_data = data[support_query_split_point:]
            support_label = label[:support_query_split_point]
            query_label = label[support_query_split_point:]

            y = this_mlp(support_data.to(torch.float32))
            loss_f = nn.CrossEntropyLoss()
            loss = loss_f(y.reshape((support_query_split_point, -1)), label_trans(support_label, 2))
            loss.backward(create_graph=True)
            this_mlp.update(lr=1e-2, parent=maml.model)
            maml_optimizer.zero_grad()

            print("#" * 100)
            y = this_mlp(query_data.to(torch.float32))
            print(y)
            y = res_trans(y.reshape(-1, 2))
            print(y)
            print(query_label)
            query_num.append(y.shape[0])
            suc_num.append(0)
            print(y.shape[0])
            print(query_label.shape[0])
            for j in range(y.shape[0]):
                if y[j] == query_label[j]:
                    suc_num[-1] = suc_num[-1] + 1
            suc_percent.append(suc_num[-1] / query_num[-1])
            print(suc_num)
            print(query_num)


def top_main():
    support = "L"

    # 小任务超参数
    inner_lr = 0.001                         # 每个任务的学习率，MAML产生的模型最终只有一次迭代机会，学习率先放大一点
    inner_epoch_num = 1

    # 元学习参数定义
    tasks_batch_size = 4                                    # 每轮训练抽取任务数量
    outer_epoch_num = 100                                   # 任务迭代次数

    tasks = Tasks2()
    tasks.get_tasks_loader(support)

    maml_loss_list = list()
    this_epoch_maml_loss = list()


    # 元学习模型定义,这里直接代表的就是F的输出f
    # MAML的网络是一样的网络，参数什么的都一样，只是个傀儡，这个网络不进行向量运算，损失函数也不一样，梯度方向不是自己的，是一批小任务共同给提供的
    # 这个傀儡每轮要把自己更新的参数给到下一批任务那里，直接赋值即可，网络结构是同一个结构，只是损失函数不一样
    # 好现在架构确定了，我的模型搭建方法不太一样，要自己写，加油！
    # 所有的网络结构参数都在这里调
    maml = MAML(init_weight=None, input_dim=197, n_hidden_1=256, output_dim=2)
    maml_lr = 0.001
    maml_optimizer = torch.optim.Adam(maml.model.parameters(), lr=maml_lr)

    # 打乱任务
    # SEED = 1
    # seed(SEED)
    shuffle(tasks.loaders)

    tasks.train_loader = tasks.loaders[:28]
    tasks.test_loader = tasks.loaders[28:]

    for outer_epoch in range(outer_epoch_num):
        # 进入训练模式
        this_epoch_maml_loss.clear()
        maml.model.train()
        print("*" * 50 +"epoch", outer_epoch + 1, "..." + "*" * 50)
        # 将任务集随机打乱
        shuffle(tasks.train_loader)
        all_these_tasks = list()
        all_these_tasks.clear()
        # 这里是在这个epoch里面把任务的batch分好
        for i in range(int(len(tasks.train_loader) / tasks_batch_size)):
            all_these_tasks.append(sample(tasks.train_loader, tasks_batch_size))
        # 遍历本次epoch里面的每一个batch
        for these_tasks in all_these_tasks:
            # 使用MAML输出与本轮batch包含任务数目相同的子模型
            these_mlp = maml.get_these_models(len(these_tasks))
            # 这里累加每一个子任务的query损失
            maml_loss = 0
            for task_id, this_task in enumerate(these_tasks):
                # 这里取到了一个任务
                print("training task", task_id + 1, "...")
                these_mlp[task_id].train()
                # MAML模型要求每个模型每次任务只support一次，query一次，将query损失累积起来作为最终损失
                for x, (data, label) in enumerate(this_task):
                    # 这里batch抽取出来，用什么模型对接就可以了，全连接层或者卷积神经网络吧，接着再往下写，用均值损失进行元更新
                    if x > 0:
                        raise "小样本任务batch出问题了"
                    data_num = data.shape[0]
                    support_query_split_point = int(data_num * 0.7)          # 支持数据与查询数据的分割点
                    support_data = data[:support_query_split_point]
                    query_data = data[support_query_split_point:]
                    support_label = label[:support_query_split_point]
                    query_label = label[support_query_split_point:]

                    y = these_mlp[task_id](support_data.to(torch.float32))
                    loss_f = nn.CrossEntropyLoss()
                    loss = loss_f(y.reshape((support_query_split_point, -1)), label_trans(support_label, 2))
                    loss.backward(create_graph=True)
                    these_mlp[task_id].update(lr=1e-2, parent=maml.model)
                    maml_optimizer.zero_grad()

                    y = these_mlp[task_id](query_data.to(torch.float32))
                    loss = loss_f(y, label_trans(query_label, 2))
                    # 在query上计算的损失不累计到各自的模型里，累加计入maml的损失当中
                    maml_loss = maml_loss + loss

            # 一批任务训练完毕，进行元更新
            maml_loss = maml_loss / tasks_batch_size
            this_epoch_maml_loss.append(maml_loss.item())
            print("maml_loss", maml_loss.item())
            maml_loss.backward(retain_graph=True)
            maml_optimizer.step()
            maml_optimizer.zero_grad()

        maml_loss_list.append(this_epoch_maml_loss.copy())

    # 绘图
    plt.figure(figsize=(12, 10))

    y = [min(loss) for loss in maml_loss_list]
    x = [(i + 1) for i in range(len(y))]


    plt.plot(x, y, color='wheat')
    plt.title(support + ' maml loss')
    plt.xlabel('epoch num')
    plt.ylabel('maml loss')
    plt.show()


    # 训练结束，进入测试环节，这里先测试吧，不随便保存模型，测试结果不错之后再保存一次
    print("*" * 100 + "test" + "*" * 100)
    query_num = list()           # 记录每个任务query data总数
    suc_num = list()             # 记录每个任务预测成功的数量
    suc_percent = list()         # 记录每个测试任务的正确率

    for i, test_loader in enumerate(tasks.test_loader):
        # 测试的时候挨个抽任务进行测试，由于我这里的主要任务还是利用元学习来进行情绪分类，要在这里测试元学习可以最终提供的准确率
        this_mlp = maml.get_these_models(1)[0]         # 生成1个就行，挨个测试，这里只测试小模型的精度，不测试maml的优度
        for x, (data, label) in enumerate(test_loader):
            if x > 0:
                print("小样本batch出问题了")
            data_num = data.shape[0]
            print(data.shape)
            support_query_split_point = int(data_num * 0.7)  # 支持数据与查询数据的分割点
            support_data = data[:support_query_split_point]
            query_data = data[support_query_split_point:]
            support_label = label[:support_query_split_point]
            query_label = label[support_query_split_point:]

            y = this_mlp(support_data.to(torch.float32))
            loss_f = nn.CrossEntropyLoss()
            loss = loss_f(y, label_trans(support_label, 2))
            loss.backward(create_graph=True)
            this_mlp.update(lr=1e-2, parent=maml.model)
            maml_optimizer.zero_grad()

            print("#" * 100)
            y = this_mlp(query_data.to(torch.float32))
            print(y)
            y = res_trans(y.reshape(-1, 2))
            print(y)
            print(query_label)
            query_num.append(y.shape[0])
            suc_num.append(0)
            print(y.shape[0])
            print(query_label.shape[0])
            for j in range(y.shape[0]):
                if y[j] == query_label[j]:
                    suc_num[-1] = suc_num[-1] + 1
            suc_percent.append(suc_num[-1] / query_num[-1])
            print(suc_num)
            print(query_num)


if __name__ == "__main__":
    top_main()