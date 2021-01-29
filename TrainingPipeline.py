# Manages training life cycle
import os

import torch
from torch import nn
from torch.optim import SGD, Adam
from torchvision import datasets, models, transforms

from Perceptron import Perceptron


class TrainingPipeline:
    def __init__(self):
        self.models = [Perceptron(16 * 16, 1, nn.LeakyReLU()), Perceptron(16 * 16, 20, nn.Sigmoid()),
                       Perceptron(16 * 16, 16 * 16, nn.Sigmoid())]
        self.loss = [nn.MSELoss(), nn.MSELoss(), nn.MSELoss()]
        self.optimizer = self.init_optimizer(3)

    def init_optimizer(self, approach):
        self.optimizer = [SGD(self.models[approach - 1].parameters(), lr=0.001),
                          Adam(self.models[approach - 1].parameters(), lr=0.001),
                          Adam(self.models[approach - 1].parameters(), lr=0.001)]

        return self.optimizer

    # Will try 3 approaches as listed in Step 2
    def get_model(self, approach):
        model = self.models[approach - 1]
        loss_fn = self.loss[approach - 1]
        optimizer = self.optimizer[approach - 1]
        return model, loss_fn, optimizer

    def train(self, x, y, model, optimizer, loss_fn, approach, num_epochs=10000):
        loss_history = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            pred = model(x)
            if approach != 3:
                pred = pred.squeeze(1)

            loss_value = loss_fn(pred, y)
            loss_value.backward()
            optimizer.step()
            loss_history.append(loss_value)

        return loss_history

    @torch.no_grad()
    def val_loss(self, x, y, model, loss_fn):
        prediction = model(x)
        val_loss = loss_fn(prediction, y)
        return val_loss.item()

    # Create training and validation datasets and initialize data loaders
    def initialize_data(self, data_dir):
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

        return image_datasets, image_dataloaders

    # get labels for Approach 2
    def get_lbl_tensor(self, image_datasets, y, kind='train'):
        mapped_lbls = [int(image_datasets[kind].classes[lookup]) for lookup in y]

        lbls_list = torch.zeros(10, 20)
        for i, indx in enumerate(mapped_lbls):
            lbls_list[i][indx - 1] = 1

        return lbls_list

    def load_all_data(self, image_dataloaders, kind='train'):
        # load
        X_train, Y_train = next(iter(image_dataloaders[kind]))
        # reshape and flatten
        X_train_f = torch.flatten(X_train[:, 0], start_dim=1)

        return X_train, Y_train, X_train_f
