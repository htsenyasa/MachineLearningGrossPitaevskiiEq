from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torch.autograd import Variable
import os.path
import time
import numpy as np

import readmnistdata as rm
import sampletrainloader as tl
import analyzer as an   
import nlse_common
from nlse_parsing import process_parsing

# Training settings
parser = process_parsing(nlse_common.archs["FCN"])
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    print("Cuda is Available")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#input_size, hidden_size, hidden2_size, num_classes = args.network_arch
input_size, hidden_size, hidden2_size, hidden3_size, num_classes = args.network_arch
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
training_len = args.training_len
test_len = args.test_len
data_file = args.data_file
label_file = args.label_file

if (args.inter_param).is_integer():
    args.inter_param = int(args.inter_param)
print("FFN running, Interaction param: {}".format(args.inter_param))

data_file = "potential-g-{}-.dat".format(args.inter_param)
label_file = "energy-g-{}-.dat".format(args.inter_param)

t = tl.nonlinear1D(data_file, label_file, training_len, test_len)
train_dataset, test_dataset = t.init_tensor_dataset()

train_loader = data_utils.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, **kwargs)
test_loader = data_utils.DataLoader(test_dataset, batch_size = args.test_batch_size, shuffle=False, **kwargs)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.fc4 = nn.Linear(hidden3_size, num_classes)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


net = Net(input_size, hidden_size, num_classes)
if args.cuda:
    net.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
res = an.analyzer(args)

def train(epoch):
    for batch_idx, (data, labels) in enumerate(train_loader):
        if args.cuda:
            data, labels = data.cuda(), labels.cuda()
        data = Variable(data)
        labels = Variable(labels).float()
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

#        if (i) % res.batch_size == 0 and args.show_progress == True:
#            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(res.cur_epoch, res.epochs, i, res.training_len // res.batch_size, loss.data[0]))
#            res.step(loss.data[0])

        if batch_idx % args.log_interval == 0 and args.show_progress == True:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

    res.step(loss.data[0])


info_file_name = "../figs/FFNTEST/" + os.path.splitext(data_file)[0]


#63.33900332450867

def test():
    for data, labels in test_loader:
        if args.cuda:
            data, labels = data.cuda(), labels.cuda()
        data = Variable(data)
        outputs = net(data)
        predicted = outputs.data.cpu().numpy()
        real = test_dataset.target_tensor.cpu().numpy()
        real = real.reshape([test_len, 1])
        res.calc_error(real, predicted)
        #res.display_plot()

#        global info_file_name
#        file_name = info_file_name + "epoch-{}-.inf".format(res.cur_epoch)
#        an.save_info(res, file_name)

    return predicted

start = time.time()

while res.cur_epoch != res.epochs + 1:
    train(res.cur_epoch)
#    if res.cur_epoch % (res.epochs / 3) == 0:
#        test()
    res.cur_epoch +=1

res.cur_epoch = res.epochs
test()

end = time.time()
print(end - start)
