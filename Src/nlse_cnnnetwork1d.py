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
import nlse_dataloader as loader
import nlse_tracer as tracer 
import nlse_common
from nlse_parsing import process_parsing
import nlse_readdata as rd

parser = process_parsing(nlse_common.archs["CNN"])
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print("Cuda is Available")
    
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

#input_size, hidden_size, hidden2_size, num_classes = args.network_arch
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
training_len = args.training_len
test_len = args.test_len

#data_filename, label_filename = nlse_common.get_filenames(args)

tl = loader.Dataloader(args.data_filename, args.label_filename, training_len, test_len, unsqueeze = False)
train_dataset, test_dataset = tl.init_tensor_dataset()
res = tracer.Tracer(args)

if args.cross_test == True:
    print("Cross Test is Enabled")
    cs_tl = loader.Dataloader(args.data_filename2, args.label_filename2, training_len, test_len, unsqueeze = False)
    cs_train_dataset, cs_test_dataset = cs_tl.init_tensor_dataset()
    test_dataset = cs_test_dataset

train_loader = data_utils.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, **kwargs)
test_loader = data_utils.DataLoader(test_dataset, batch_size = args.test_batch_size, shuffle=False, **kwargs)

class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv1d(2, 10, 2)
        self.conv2 = nn.Conv1d(5, 20, 2)
        self.conv3 = nn.Conv1d(10, 20, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(310, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 1)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

res.chrono_point("model_init_start")

model = CnnNet()
if args.cuda:
    model.cuda()
criterion = F.mse_loss
optimizer = optim.Adam(model.parameters(), lr = args.lr)

if args.load_state == True:
    inf = tracer.load_info(args.info_file)
    tracer.update_state(model, optimizer, inf, res, *tracer.get_state(args.load_state_file))


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and (args.display_progress == True):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    res.step(loss.data[0])

print(args.info_file)

def test():
    model.eval()
    test_loss = 0
    correct = 0
    predicted = np.empty((test_len, 1), float)
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile = True), Variable(target).float()
        outputs = model(data)
        predicted[batch_idx * test_loader.batch_size: (batch_idx + 1) * test_loader.batch_size] = outputs.data.cpu().numpy()
    
    real = test_dataset.target_tensor.cpu().numpy()
    real = real.reshape([test_len, 1])
    res.analyze(real, predicted)

    if not args.test_case:
        #res.plot_figure()
        tracer.save_info(res, args.info_file)

    return predicted

res.chrono_point("model_init_end")

res.chrono_point("train_start")

while res.cur_epoch != res.epochs + 1:
    train(res.cur_epoch)
#    if res.cur_epoch % (res.epochs / 3) == 0 and not args.test_case:
#        test()
    res.cur_epoch +=1
    

#res.cur_epoch = res.epochs
res.chrono_point("train_end")
test()

tracer.save_state(model, optimizer, res.cur_epoch, args.save_state_file)

#res.plot_loss()
print(res.chrono_points["train_end"] - res.chrono_points["train_start"])
print(res.chrono_points)









#class CnnNet(nn.Module):
#    def __init__(self):
#        super(CnnNet, self).__init__()
#        self.conv1 = nn.Conv1d(1, 40, 2)
#        self.conv2 = nn.Conv1d(20, 40, 2)
#        self.conv3 = nn.Conv1d(20, 40, 2)
#        self.relu = nn.ReLU()   
#        self.fc1 = nn.Linear(300, 150)
#        self.fc2 = nn.Linear(150, 100)
#        self.fc3 = nn.Linear(100, 50)
#        self.fc4 = nn.Linear(50, 20)
#        self.fc5 = nn.Linear(20, 1)
#
#
#    def forward(self, x):
#        activation = F.relu
#        x = activation(F.max_pool2d(self.conv1(x), 2))
##        print(x.shape)
#        x = activation(F.max_pool2d(self.conv2(x), 2))
##        print(x.shape)
#        x = activation(F.max_pool2d(self.conv3(x), 2))
##        print(x.shape)
#        x = x.view(x.size(0), -1)
##        print(x.shape)
#        x = self.fc1(x)
#        x = self.relu(x)
#        x = self.fc2(x)
#        x = self.relu(x)
#        x = self.fc3(x)
#        x = self.relu(x)
#        x = self.fc4(x)
#        x = self.relu(x)
#        x = self.fc5(x)
#        return x

#class CnnNet(nn.Module):
#    def __init__(self):
#        super(CnnNet, self).__init__()
#        self.conv1 = nn.Conv1d(1, 20, 2)
#        self.conv2 = nn.Conv1d(10, 20, 2)
#        self.conv3 = nn.Conv1d(10, 20, 2)
#        self.relu = nn.ReLU()   
#        self.fc1 = nn.Linear(150, 125)
#        self.fc2 = nn.Linear(125, 100)
#        self.fc3 = nn.Linear(100, 50)
#        self.fc4 = nn.Linear(50, 20)
#        self.fc5 = nn.Linear(20, 1)
#
#
#    def forward(self, x):
#        x = F.relu(F.max_pool2d(self.conv1(x), 2))
#        x = F.relu(F.max_pool2d(self.conv2(x), 2))
#        x = F.relu(F.max_pool2d(self.conv3(x), 2))
#        x = x.view(x.size(0), -1)
#        x = self.fc1(x)
#        x = self.relu(x)
#        x = self.fc2(x)
#        x = self.relu(x)
#        x = self.fc3(x)
#        x = self.relu(x)
#        x = self.fc4(x)
#        x = self.relu(x)
#        x = self.fc5(x)
#        return x