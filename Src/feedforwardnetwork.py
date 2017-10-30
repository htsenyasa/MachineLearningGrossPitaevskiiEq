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
import analyze as an

# Training settings
parser = argparse.ArgumentParser(description='Fully Connected FeedForwardNetwork for nonlinearSE')

parser.add_argument('--batch-size',        type=int,             default=10,                 metavar='N',    help = 'input batch size for training (default: 64)')
parser.add_argument('--test-batch-size',   type=int,             default=1000,               metavar='N',    help = 'input batch size for testing (default: 1000)')
parser.add_argument('--epochs',            type=int,             default=50,                 metavar='N',    help = 'number of epochs to train (default: 10)')
parser.add_argument('--lr',                type=float,           default=0.001,              metavar='LR',   help = 'learning rate (default: 0.01)')
parser.add_argument('--momentum',          type=float,           default=0.5,                metavar='M',    help = 'SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda',           action='store_true',  default=False,                              help = 'disables CUDA training')
parser.add_argument('--seed',              type=int,             default=1,                  metavar='S',    help = 'random seed (default: 1)')
parser.add_argument('--log-interval',      type=int,             default=10,                 metavar='N',    help = 'how many batches to wait before logging training status')
parser.add_argument('--network-arch',      type=int,             default=[256, 40, 20, 1],   nargs='+',      help = 'Network arch : (default: 256-40-20-1)')
parser.add_argument('--training-len',      type=int,             default=3500,                               help = 'Training len (default: 3500)')
parser.add_argument('--test-len',          type=int,             default=500,                                help = 'Test len (default: 500)')
parser.add_argument('--runtime-count',     type=int,             default=0,                                  help = 'this parameter counts that how many times the program is runned')
parser.add_argument('--show-progress',     type=bool,            default=False,                              help = 'display progress (default:False)')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

input_size, hidden_size, hidden2_size, num_classes = args.network_arch
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
training_len = args.training_len
test_len = args.test_len

t = tl.nonlinear1D(training_len, test_len)
train_dataset, test_dataset = t.init_tensor_dataset()

train_loader = data_utils.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, **kwargs)
test_loader = data_utils.DataLoader(test_dataset, batch_size = args.test_batch_size, shuffle=False, **kwargs)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

net = Net(input_size, hidden_size, num_classes)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = Variable(images)
        labels = Variable(labels).float()

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % batch_size == 0 and args.show_progress == True:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))


#Test the Model
for targets, labels in test_loader:
    targets = Variable(targets)
    outputs = net(targets)
    predicted = outputs.data


a = predicted.numpy()
b = test_dataset.target_tensor.numpy()
res = an.analyze(b, a, args)
res.display_plot()

torch.save(net.state_dict(), 'model.pkl')
