from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import readmnistdata as rm
import sampletrainloader as tl
import analyzer as an
import os.path

# Training settings
parser = argparse.ArgumentParser(description='CNN for nonlinearSE')

parser.add_argument('--batch-size',        type=int,             default=30,                 metavar='N',    help = 'input batch size for training (default: 64)')
parser.add_argument('--test-batch-size',   type=int,             default=1500,               metavar='N',    help = 'input batch size for testing (default: 1000)')
parser.add_argument('--epochs',            type=int,             default=20,                 metavar='N',    help = 'number of epochs to train (default: 10)')
parser.add_argument('--lr',                type=float,           default=1e-3,               metavar='LR',   help = 'learning rate (default: 0.01)')
parser.add_argument('--momentum',          type=float,           default=0.2,                metavar='M',    help = 'SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda',           action='store_true',  default=False,                              help = 'disables CUDA training')
parser.add_argument('--seed',              type=int,             default=1,                  metavar='S',    help = 'random seed (default: 1)')
parser.add_argument('--log-interval',      type=int,             default=10,                 metavar='N',    help = 'how many batches to wait before logging training status')
parser.add_argument('--network-arch',      type=str,             default="conv1d",           nargs='+',      help = 'Network arch : (default: 256-40-20-1)')
parser.add_argument('--training-len',      type=int,             default=8500,                               help = 'Training len (default: 3500)')
parser.add_argument('--test-len',          type=int,             default=1500,                               help = 'Test len (default: 500)')
parser.add_argument('--runtime-count',     type=int,             default=0,                                  help = 'this parameter counts that how many times the program is runned')
parser.add_argument('--show-progress',     action='store_true',  default=False,                              help = 'display progress (default:False)')
parser.add_argument('--data-file',         type=str,             default="potential-g-20-.dat",              help = 'data file to read (default = "potential.dat")')
parser.add_argument('--label-file',        type=str,             default="energy-g-20-.dat",                 help = 'label file to read (default = "energy.dat")')
parser.add_argument('--inter-param',       type=float,           default=0.0,                                  help = 'interaction parameter program uses this parameter to choose which file to open (default: 0)')




args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(1)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#input_size, hidden_size, hidden2_size, num_classes = args.network_arch
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
training_len = args.training_len
test_len = args.test_len
#data_file = args.data_file
#label_file = args.label_file

if (args.inter_param).is_integer():
    args.inter_param = int(args.inter_param)

print("CNN running, Interaction param: {}".format(args.inter_param))


data_file = "potential-g-{}-.dat".format(args.inter_param)
label_file = "energy-g-{}-.dat".format(args.inter_param)


t = tl.nonlinear1D(data_file, label_file, training_len, test_len, cnn = True)
train_dataset, test_dataset = t.init_tensor_dataset()

train_loader = data_utils.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, **kwargs)
test_loader = data_utils.DataLoader(test_dataset, batch_size = args.test_batch_size, shuffle=False, **kwargs)


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, 2)
        self.conv2 = nn.Conv1d(5, 20, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(310, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 1)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = CnnNet()
criterion = F.mse_loss
optimizer = optim.Adam(model.parameters(), lr = args.lr)
res = an.analyzer(args)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and (args.show_progress == True):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    res.step(loss.data[0])


info_file_name = "../figs/CNN/" + os.path.splitext(data_file)[0]

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile = True), Variable(target).float()
        outputs = model(data)
        predicted = outputs.data.numpy()
        real = test_dataset.target_tensor.numpy()
        real = real.reshape([test_len, 1])
        res.calc_error(real, predicted)
        #res.display_plot()

        global info_file_name
        file_name = info_file_name + "conv1d-epoch-{}-.inf".format(res.cur_epoch)
        an.save_info(res, file_name)

    return predicted


while res.cur_epoch != res.epochs + 1:
    train(res.cur_epoch)
    if res.cur_epoch % (res.epochs / 3) == 0:
        test()
    res.cur_epoch +=1

res.cur_epoch = res.epochs
