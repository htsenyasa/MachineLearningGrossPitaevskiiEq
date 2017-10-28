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

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help = 'input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help = 'input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help = 'number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help = 'learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help = 'SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help = 'disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help = 'random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help = 'how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


# Hyper Parameters
input_size = 256
hidden_size = 30
hidden2_size = 20
num_classes = 1
num_epochs = 50
batch_size = args.batch_size
learning_rate = 0.001
training_len = 3500
test_len = 500

t = tl.nonlinear1D(training_len, test_len)
pixel_tensor, label_tensor = t.get_data()
test_tensor, test_label_tensor = t.get_data(train = False)

train_dataset = data_utils.TensorDataset(pixel_tensor, label_tensor)
test_dataset = data_utils.TensorDataset(test_tensor, test_label_tensor)

train_loader = data_utils.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, **kwargs)
test_loader = data_utils.DataLoader(test_dataset, batch_size = args.test_batch_size, shuffle=False, **kwargs)

# Neural Network Model (1 hidden layer)
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

        if (i+1) % batch_size == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))


#Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = net(images)
    predicted = outputs.data


a = predicted.numpy()
b = test_label_tensor.numpy();
error = np.array([abs((a[i] - b[i])**2) for i in range(len(test_label_tensor))])

plt.plot(b,b, "--r", label="real data")
plt.plot(b, a, ".")
plt.xlabel("Real")
plt.ylabel("Predicted")
plt.legend()

error = sum(error)/len(test_label_tensor)
print(error)
plt.figtext(0.6, 0.2, "256-30-20-1\nlr={}\nepoch={}\ntrain_len={}\ntest_len={}\nerror={}".format(learning_rate, num_epochs, training_len, test_len, error))


plt.savefig("plot001.png", dip=500)
plt.grid()
plt.show()

torch.save(net.state_dict(), 'model.pkl')
