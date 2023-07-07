import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

# 加载训练数据
training_data = datasets.FashionMNIST(
    root="data/fashion",
    train=True,
    download=True,
    transform=ToTensor()
)

# 加载预测数据
test_data = datasets.FashionMNIST(
    root="data/fashion",
    train=False,
    download=True,
    transform=ToTensor()
)

# 分类
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# 随机展示训练数据中的几张图片
def show_data():
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        print(f"sample_idx: {sample_idx}")
        print(f"data: {training_data[sample_idx]}")
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

# 创建神经网络
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # 使用 nn.Flatten 层，将每个 2D 28x28 图像转换为784个像素值的连续数组
        self.flatten = nn.Flatten()
        # Sequential是一个有序的模块容器，数据以定义的相同顺序通过所有的模块
        self.linear_relu_stack = nn.Sequential(
            # 线性层是使用其存储的权重和偏置对输入进行线性变换的模块。
            nn.Linear(28*28, 512),
            # 非线性激活函数帮助模型在输入和输出之间创建复杂的映射关系。它们在线性变换后使用，以引入非线性，帮助神经网络学习各种现象。
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # 对数据进行操作
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 训练循环
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 计算预测和损失
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} batch:{batch:>5} [{current:>5d}/{size:>5d}]")

# 测试循环
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train_data():
    # 创建模型
    model = NeuralNetwork()
    # 在每个批次/每次迭代更新模型参数的程度
    learning_rate = 1e-3
    # 在更新参数之前，通过网络传播的数据样本的数量
    batch_size = 64
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 创建优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # 迭代数据集的次数
    epochs = 10

    train_dataloader = DataLoader(training_data, batch_size)
    test_dataloader = DataLoader(test_data, batch_size)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    # 训练好的模型将模型参数保存起来
    torch.save(model.state_dict(), "data/fashion/model/model.pt")
    print("Done!")


def showOne(img, i):
    plt.title(labels_map[i])
    plt.imshow(img, cmap="gray")
    plt.show()

# 预测
def eval():
    # 从测试集中随机获取一个下标
    sample_idx = torch.randint(len(test_data), size=(1,)).item()
    # print(f"sample_idx: {sample_idx}")
    # 获取图片以及分类标签
    img, label = test_data[sample_idx]
    print(f"label:{label}")
    print(f"img:{img.shape}")

    # 创建模型
    model = NeuralNetwork()
    # 加载模型
    model.load_state_dict(torch.load("data/fashion/model/model.pt"))
    # 关闭梯度计算，提升预测速度
    with torch.no_grad():
        # 模型预测
        p = model(img)
        # 获取最大概率位置
        _, predicted = torch.max(p.data, 1)
        print(f"value:{predicted}")
        showOne(img[0], predicted.item())


if __name__ == '__main__':
    #show_data()
    train_data()
    #eval()
