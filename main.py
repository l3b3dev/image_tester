import os

import torch
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torch.nn as nn
import numpy as np
from torch.optim import SGD

import matplotlib.pyplot as plt


class Perceptron(nn.Module):
    def __init__(self, net_inputs, net_output):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(net_inputs, net_output)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return x


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_dir = "data"

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ]),
    }

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'val']}
    # Create training and validation dataloaders
    image_dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=image_datasets[x].__len__(), shuffle=True)
        for x
        in
        ['train', 'val']}

    # load
    X_train, Y_train = next(iter(image_dataloaders['train']))
    # reshape and flatten
    X_train_f = torch.flatten(X_train[:, 0], start_dim=1)

    # plot train data with labels
    R, C = 1, 10
    fig, ax = plt.subplots(R, C)
    fig.suptitle('Training Data with corresponding labels')
    for i, plot_cell in enumerate(ax):
        plot_cell.grid(False)
        plot_cell.axis('off')
        plot_cell.set_title(image_datasets['train'].classes[Y_train[i].item()])
        plot_cell.imshow(X_train[i][0], cmap='gray')
        # plot_cell.imshow( X_train_f[i].reshape(16,16), cmap='gray')
    plt.tight_layout()
    plt.show()

    # # train model 1
    # model1 = Perceptron(16*16,1)
    # loss_func = nn.MSELoss()
    #
    # opt = SGD(model1.parameters(), lr=0.001)
    #
    # loss_history = []
    # for epoch in range(100):
    #     opt.zero_grad()
    #     loss_value = loss_func(model1(X_train_f), torch.FloatTensor( [float(image_datasets['train'].classes[lookup]) for lookup in Y_train]))
    #     loss_value.backward()
    #     opt.step()
    #     loss_history.append(loss_value)
    #
    # plt.plot(loss_history)
    # plt.title('Loss variation over increasing epochs')
    # plt.xlabel('epochs')
    # plt.ylabel('loss value')
    # plt.show()
    #
    #
    # X_test = X_train_f[3]
    # model1.eval()
    # y_pred = model1(X_test)
    #
    # # plot predicted data with
    # R, C = 1, 2
    # fig, (ax1,ax2) = plt.subplots(R, C)
    # fig.suptitle('Predicted Data')
    # #for i, plot_cell in enumerate(ax):
    # ax1.grid(False)
    # ax1.axis('off')
    # ax1.set_title('Actual')
    # ax1.imshow(X_train[0][0], cmap='gray')
    #
    # ax2.grid(False)
    # ax2.axis('off')
    # ax1.set_title('Predicted')
    # ax2.imshow(X_train[0][0], cmap='gray')
    #
    # plt.tight_layout()
    # plt.show()

    # train model 2
    # model2 = Perceptron(16 * 16, 20)
    # loss_func = nn.MSELoss()
    #
    # opt = SGD(model2.parameters(), lr=0.001)
    # mapped_lbls = [int(image_datasets['train'].classes[lookup]) for lookup in Y_train]
    #
    # lbls_list = torch.zeros(10, 20)
    # for i, indx in enumerate(mapped_lbls):
    #     lbls_list[i][indx-1] = 1
    #
    # loss_history = []
    # for epoch in range(100):
    #     opt.zero_grad()
    #     loss_value = loss_func(model2(X_train_f), lbls_list)
    #     loss_value.backward()
    #     opt.step()
    #     loss_history.append(loss_value)
    #
    # plt.plot(loss_history)
    # plt.title('Loss variation over increasing epochs')
    # plt.xlabel('epochs')
    # plt.ylabel('loss value')
    # plt.show()
    #
    # X_test = X_train_f[0]
    # model2.eval()
    # vv = model2(X_test)
    # y_pred = torch.argmax(model2(X_test))
    # p_val = y_pred.item()+1

    # # plot predicted data with
    # R, C = 1, 2
    # fig, (ax1, ax2) = plt.subplots(R, C)
    # fig.suptitle('Predicted Data')
    # # for i, plot_cell in enumerate(ax):
    # ax1.grid(False)
    # ax1.axis('off')
    # ax1.set_title('Actual')
    # ax1.imshow(X_train[0][0], cmap='gray')
    #
    # ax2.grid(False)
    # ax2.axis('off')
    # ax1.set_title('Predicted')
    # ax2.imshow(X_train[0][0], cmap='gray')
    #
    # plt.tight_layout()
    # plt.show()
    #

    model3 = Perceptron(16 * 16, 16 * 16)
    loss_func = nn.MSELoss()

    opt = SGD(model3.parameters(), lr=0.001)

    loss_history = []
    for epoch in range(100):
        opt.zero_grad()
        loss_value = loss_func(model3(X_train_f), X_train_f)
        loss_value.backward()
        opt.step()
        loss_history.append(loss_value)

    plt.plot(loss_history)
    plt.title('Loss variation over increasing epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss value')
    plt.show()

    X_test = X_train_f[0]
    model3.eval()
    Y_test_pred = model3(X_test)
    ff = X_train_f[0]
    asdf = Y_test_pred.reshape(16, 16)

    # # plot predicted data with
    R, C = 1, 2
    fig, (ax1, ax2) = plt.subplots(R, C)
    fig.suptitle('Predicted Data')
    # for i, plot_cell in enumerate(ax):
    ax1.grid(False)
    ax1.axis('off')
    ax1.set_title('Actual')
    ax1.imshow(X_train[0][0], cmap='gray')

    ax2.grid(False)
    ax2.axis('off')
    ax2.set_title('Predicted')
    ax2.imshow(asdf.detach().numpy(), cmap='gray')

    plt.tight_layout()
    plt.show()

