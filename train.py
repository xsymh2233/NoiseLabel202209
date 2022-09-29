import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from LeNet5 import *
import loss1
import numpy as np

train_batch_size=12

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    print('cuda')
else:
    device = torch.device('cpu')


# 定义超参数
noise_fractions=0.32
batch_size = 64  # 一次训练的样本数目
learning_rate = 0.0001  # 学习率
iteration_num = 50  # 迭代次数
network = LeNet5()  # 实例化网络
optimizer = optim.SGD(params=network.parameters(), lr=learning_rate, momentum=0.78)
bootstarp_beta = 0.95
# 定义损失函数
criterion = loss1.Bootstrapping(10, t=1 - bootstarp_beta)
#criterion=nn.CrossEntropyLoss()


def get_data():   # 仅参考用
    """获取数据"""
    # 获取测试集
    train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),  # 转换成张量
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                       ]))
    train_loader = DataLoader(train, batch_size=batch_size)  # 分割测试集

    # 获取测试集
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),  # 转换成张量
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                      ]))
    test_loader = DataLoader(test, batch_size=batch_size)  # 分割训练

    # 返回分割好的训练集和测试集
    return train_loader, test_loader

def get_traindata_noise():
    """获取数据"""
    train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),  # 转换成张量
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                       ]))
    train_loading = DataLoader(train, batch_size=len(train))  # 分割训练集

    for data in train_loading:
        x, y = data
    y_noise = torch.ones(len(y))

    for j in range(len(y)):
        rd = np.random.rand()
        if rd <= 1 - noise_fractions:
            y_noise[j] = y[j]
        else:
            for i in range(10):
                if 1 - noise_fractions + i * noise_fractions < rd <= 1 - noise_fractions + (i + 1) * noise_fractions:
                    y_noise[j] *= i    # 加标签噪声
    # print(y_noise)
    # print(y)
    train_noise = TensorDataset(x, y_noise)    # 装包
    train_noise_loader = DataLoader(train_noise, batch_size=batch_size)  # 分割训练集
    train_loader = DataLoader(train_noise, batch_size=batch_size)  # 分割训练集
    # 返回分割好的训练集
    return train_noise_loader, train_loader

def get_testdata_noise():
    # 获取测试集
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),  # 转换成张量
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                      ]))
    test_loading = DataLoader(test, batch_size=len(test))  # 分割测试集

    for data in test_loading:
        x, y = data
    y_noise = torch.ones(len(y))

    for j in range(len(y)):
        rd = np.random.rand()
        if rd <= 1 - noise_fractions:
            y_noise[j] = y[j]
        else:
            for i in range(10):
                if 1 - noise_fractions + i * noise_fractions < rd <= 1 - noise_fractions + (i + 1) * noise_fractions:
                    y_noise[j] *= i  # 加标签噪声

    test_noise = TensorDataset(x, y_noise)  # 装包
    #test_noise_loader = DataLoader(test_noise, batch_size=batch_size)  # 分割测试集
    test_loader = DataLoader(test, batch_size=batch_size)  # 分割测试集

    # 返回分割好的测试集
    return  test_loader


def train(model, epoch, train_loader):
    """训练"""

    # 训练模式
    model=model.cuda()
    model.train()

    # 迭代
    for step, (x, y1) in enumerate(train_loader):
        y=y1.long()
        x = x.cuda()
        y = y.cuda()
        output = model(x)
        # 计算损失

        loss = criterion(output,y)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新梯度
        optimizer.step()

        # 打印损失
        if step % 100 == 0:
            print('Epoch: {}, Step {}, Loss: {}'.format(epoch, step, loss))


def test(model, test_loader):
    """测试"""

    # 测试模式
    model = model.cuda()
    model.eval()

    # 存放正确个数
    correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.cuda()
            y = y.cuda()

            # 获取结果
            output = model(x)

            # 预测结果
            pred = output.argmax(dim=1, keepdim=True)

            # 计算准确个数
            correct += pred.eq(y.view_as(pred)).sum().item()

    # 计算准确率
    accuracy = correct / len(test_loader.dataset) * 100

    # 输出准确
    print("Test Accuracy: {}%".format(accuracy))


def main():

    # 获取数据
    trainND ,trainD = get_traindata_noise()
    testD = get_testdata_noise()


    #迭代
    for epoch in range(iteration_num):
        print("\n================ epoch: {} ================".format(epoch))
        train(network, epoch, trainND)
        test(network, testD)


if __name__ == "__main__":
    main()