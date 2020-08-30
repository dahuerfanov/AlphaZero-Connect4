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

from CategoricalCrossEntropy import CategoricalCrossEntropy
from constants import LR_SGD, MOMENTUM_SGD, WD_SGD, ROWS, COLS, BATCH, EPOCHS


class NNet(Module):

    def __init__(self, name):
        super(NNet, self).__init__()

        self.name = name
        self.cnn_layers = Sequential(
            Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=64),
            ReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=64),
            ReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=64),
            ReLU(inplace=True),
        )

        self.v_layers = Sequential(
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(num_features=1),
            ReLU(inplace=True),
            Flatten(),
            Linear(in_features=ROWS * COLS, out_features=32),
            Dropout(p=0.3, inplace=True),
            Linear(in_features=32, out_features=1),
            Tanh()
        )

        self.p_layers = Sequential(
            Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(num_features=2),
            ReLU(inplace=True),
            Flatten(),
            Linear(in_features=2 * ROWS * COLS, out_features=COLS),
            Softmax(dim=1),
        )

        self.criterion_p = CategoricalCrossEntropy()
        self.criterion_v = MSELoss()
        if torch.cuda.is_available():
            self.criterion_p = self.criterion_p.cuda()
            self.criterion_v = self.criterion_v.cuda()

        self.optimizer = SGD(self.parameters(), lr=LR_SGD, momentum=MOMENTUM_SGD,
                             nesterov=True, weight_decay=WD_SGD)

    def forward(self, x):
        x0 = self.cnn_layers(x)
        x1 = self.v_layers(x0)
        x2 = self.p_layers(x0)
        return x1, x2

    def train_batch(self, x_train, v_train, p_train):

        self.train()

        x_train = Variable(x_train, requires_grad=True)
        v_train = Variable(v_train, requires_grad=True)
        p_train = Variable(p_train, requires_grad=True)

        # ========forward pass=====================================
        v_output_train, p_output_train = self(x_train)

        # clearing the Gradients of the model parameters
        self.optimizer.zero_grad()

        v_loss_train = self.criterion_v(v_output_train.reshape(list(v_output_train.size())[0]), v_train)
        p_loss_train = self.criterion_p(p_output_train, p_train)
        loss_train = v_loss_train + p_loss_train

        # computing the updated weights of all the model parameters
        loss_train.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return v_loss_train.item(), p_loss_train.item(), \
               (torch.round(v_output_train.reshape(list(v_output_train.size())[0])) == v_train).float().sum().item()

    def validate(self, x, v, p):

        self.eval()

        # ========forward pass=====================================
        with torch.no_grad():
            v_output, p_output = self(x)

        v_loss = self.criterion_v(v_output.reshape(list(v_output.size())[0]), v)
        p_loss = self.criterion_p(p_output, p)

        return v_loss.item(), p_loss.item(), \
               (torch.round(v_output.reshape(list(v_output.size())[0])) == v).float().sum().item()

    def run(self, X, Y_v, Y_p):

        train_x, val_x, train_v, val_v, train_p, val_p = train_test_split(X, Y_v, Y_p, test_size=0.2)

        # converting the data into GPU format
        if torch.cuda.is_available():
            train_x = train_x.cuda()
            val_x = val_x.cuda()
            train_v = train_v.cuda()
            val_v = val_v.cuda()
            train_p = train_p.cuda()
            val_p = val_p.cuda()
            print("Using GPU!", torch.cuda.get_device_name(None))
        else:
            print("Using CPU :(")

        v_train_losses = []
        p_train_losses = []
        v_val_losses = []
        p_val_losses = []

        v_train_acc = []
        v_val_acc = []

        # training the model
        n_batches = len(train_x) // BATCH
        for _ in range(EPOCHS):
            loss_v_tr, loss_p_train, loss_v_val, loss_p_val = 0, 0, 0, 0
            acc_v_tr, acc_v_val = 0, 0
            div = 0
            for i in range(n_batches):
                if (i + 1) * BATCH > len(train_x):
                    break
                div = i + 1
                loss_v_tr_, loss_p_train_, acc_v_tr_ = \
                    self.train_batch(train_x[i * BATCH:(i + 1) * BATCH, ],
                                     train_v[i * BATCH:(i + 1) * BATCH, ],
                                     train_p[i * BATCH:(i + 1) * BATCH, ])
                loss_v_tr += loss_v_tr_
                loss_p_train += loss_p_train_

                acc_v_tr += acc_v_tr_

            loss_v_val, loss_p_val, acc_v_val = self.validate(val_x, val_v, val_p)

            v_train_losses.append(loss_v_tr / div)
            p_train_losses.append(loss_p_train / div)
            v_val_losses.append(loss_v_val)
            p_val_losses.append(loss_p_val)

            v_train_acc.append(acc_v_tr / (BATCH * div))
            v_val_acc.append(acc_v_val / len(val_x))

        # Plot history: loss out_v, out_v_accuracy
        plt.plot(v_train_losses, label='V loss (training data)')
        plt.plot(v_val_losses, label='V loss (validation data)')
        plt.plot(p_train_losses, label='P loss (training data)')
        plt.plot(p_val_losses, label='P loss (validation data)')
        plt.title('loss functions')
        plt.ylabel('Loss')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()

        # Plot history: loss out_dist,out_dist_accuracy
        plt.plot(v_train_acc, label='V accuracy (training data)')
        plt.plot(v_val_acc, label='V accuracy (validation data)')
        plt.title('Accuracy functions')
        plt.ylabel('Accuracy')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()
