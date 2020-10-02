import matplotlib.pyplot as plt
# PyTorch libraries and modules
import torch
# for creating validation set
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
# from torch.autograd import Variable
from torch.nn import Linear, ReLU, Sequential, Conv2d, Module, Softmax, BatchNorm2d, \
    Dropout, Flatten, Tanh, MSELoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from CategoricalCrossEntropy import CategoricalCrossEntropy
from constants import LR_SGD, MOMENTUM_SGD, WD_SGD, ROWS, COLS, BATCH, EPOCHS


class NNet(Module):

    def __init__(self, name, device):
        super(NNet, self).__init__()

        self.name = name
        self.device = device
        self.cnn_layers = Sequential(
            Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=128, momentum=0.1),
            ReLU(inplace=True),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=128, momentum=0.1),
            ReLU(inplace=True),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=128, momentum=0.1),
            ReLU(inplace=True)
        )

        self.v_layers = Sequential(
            Conv2d(in_channels=128, out_channels=4, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(num_features=4, momentum=0.1),
            ReLU(inplace=True),
            Flatten(),
            Linear(in_features=4 * ROWS * COLS, out_features=32),
            Dropout(p=0.3, inplace=True),
            ReLU(inplace=True),
            Linear(in_features=32, out_features=1),
            Tanh()
        )

        self.p_layers = Sequential(
            Conv2d(in_channels=128, out_channels=8, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(num_features=8, momentum=0.1),
            ReLU(inplace=True),
            Flatten(),
            Linear(in_features=8 * ROWS * COLS, out_features=COLS),
            Softmax(dim=1),
        )

        self.criterion_p = CategoricalCrossEntropy()
        self.criterion_v = MSELoss()

        self = self.to(device)
        self.criterion_p = self.criterion_p.to(device)
        self.criterion_v = self.criterion_v.to(device)

        self.optimizer = SGD(self.parameters(), lr=LR_SGD, momentum=MOMENTUM_SGD,
                             nesterov=True, weight_decay=WD_SGD)


    def forward(self, x):
        x0 = self.cnn_layers(x)
        x1 = self.v_layers(x0)
        x2 = self.p_layers(x0)
        return x1, x2

    def predict(self, s):
        self.eval()
        with torch.no_grad():
            x = torch.stack([s])
            # converting the data into GPU format (if available)
            x = x.to(self.device)
            x0 = self.cnn_layers(x)
            x1 = self.v_layers(x0)
            x2 = self.p_layers(x0)
            return x1[0].item(), x2[0].numpy()

    def train_batch(self, x_train, v_train, p_train):

        self.train()
        # clearing the Gradients of the model parameters
        self.optimizer.zero_grad()

        # ========forward pass=====================================
        v_output_train, p_output_train = self(x_train)

        v_loss_train = self.criterion_v(v_output_train.reshape(list(v_output_train.size())[0]), v_train)
        p_loss_train = self.criterion_p(p_output_train, p_train)
        loss_train = v_loss_train + p_loss_train

        # computing the updated weights of all the model parameters
        loss_train.backward()
        self.optimizer.step()

        return v_loss_train.item(), p_loss_train.item(), \
               (torch.round(v_output_train.reshape(list(v_output_train.size())[0])) == torch.round(
                   v_train)).float().sum().item()

    def validate(self, x, v, p):

        self.eval()
        # ========forward pass=====================================
        with torch.no_grad():
            v_output, p_output = self(x)
            v_loss = self.criterion_v(v_output.reshape(list(v_output.size())[0]), v)
            p_loss = self.criterion_p(p_output, p)

            return v_loss.item(), p_loss.item(), \
                   (torch.round(v_output.reshape(list(v_output.size())[0])) == torch.round(v)).float().sum().item()

    def run(self, X, Y_v, Y_p):
        sim_data = []
        for i in range(len(X)):
            sim_data.append([X[i], Y_v[i], Y_p[i]])
        trainset, testset = train_test_split(sim_data, test_size=0.2)

        trainloader = DataLoader(dataset=trainset, batch_size=BATCH, shuffle=True)
        testloader = DataLoader(dataset=testset, batch_size=BATCH, shuffle=True)

        v_train_losses = []
        p_train_losses = []
        v_val_losses = []
        p_val_losses = []

        v_train_acc = []
        v_val_acc = []

        # training the model
        for _ in range(EPOCHS):
            loss_v_tr, loss_p_train, loss_v_val, loss_p_val = 0, 0, 0, 0
            acc_v_tr, acc_v_val = 0, 0
            div_train = 0
            total_train = 0
            for batch_idx, (S, V, P) in enumerate(trainloader):
                S, V, P = S.to(self.device).requires_grad_(), V.to(self.device).requires_grad_(), P.to(self.device).requires_grad_()
                div_train += 1
                total_train += V.size(0)
                loss_v_tr_, loss_p_train_, acc_v_tr_ = self.train_batch(S, V, P)
                loss_v_tr += loss_v_tr_
                loss_p_train += loss_p_train_
                acc_v_tr += acc_v_tr_

            div_val = 0
            total_val = 0
            for batch_idx, (S, V, P) in enumerate(testloader):
                S, V, P = S.to(self.device).requires_grad_(False), V.to(self.device).requires_grad_(False), P.to(self.device).requires_grad_(False)
                div_val += 1
                total_val += V.size(0)
                loss_v_val_, loss_p_val_, acc_v_val_ = self.validate(S, V, P)
                loss_v_val += loss_v_val_
                loss_p_val += loss_p_val_
                acc_v_val += acc_v_val_

            v_train_losses.append(loss_v_tr / div_train)
            p_train_losses.append(loss_p_train / div_train)
            v_val_losses.append(loss_v_val / div_val)
            p_val_losses.append(loss_p_val / div_val)

            v_train_acc.append(acc_v_tr / total_train)
            v_val_acc.append(acc_v_val / total_val)

        plt.plot(p_train_losses, label='P loss (training data)')
        plt.plot(p_val_losses, label='P loss (validation data)')
        plt.title('loss functions P')
        plt.ylabel('Loss')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()

        plt.plot(v_train_losses, label='V loss (training data)')
        plt.plot(v_val_losses, label='V loss (validation data)')
        plt.title('loss functions V')
        plt.ylabel('Loss')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()

        plt.plot(v_train_acc, label='V accuracy (training data)')
        plt.plot(v_val_acc, label='V accuracy (validation data)')
        plt.title('Accuracy functions')
        plt.ylabel('Accuracy')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()