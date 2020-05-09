import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
use_gpu = torch.cuda.is_available()
print('GPU: ', use_gpu)
import Input
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
input_size = 300
import torch.nn as nn


class CNN(torch.nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        self.Conv2d2 = torch.nn.Conv2d(1, 120, (2, input_size), stride=1)
        self.Conv2d3 = torch.nn.Conv2d(1, 120, (3, input_size), stride=1)
        self.Conv2d4 = torch.nn.Conv2d(1, 120, (4, input_size), stride=1)
        #self.Conv2d5 = torch.nn.Conv2d(1, 20, (5, input_size))
        #self.Conv2d6 = torch.nn.Conv2d(1, 20, (6, input_size))
        self.fc1 = torch.nn.Linear(360, 8)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print("Linear init")
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                print("Convd2d init")
                nn.init.kaiming_normal_(m.weight)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.SM = torch.nn.Softmax(dim=1)

    def forward(self, x):
        Conv2 = torch.relu(self.Conv2d2(x)).squeeze(3)
        Conv3 = torch.relu(self.Conv2d3(x)).squeeze(3)
        Conv4 = torch.relu(self.Conv2d4(x)).squeeze(3)
        #Conv5 = torch.nn.functional.sigmoid(self.Conv2d5(x)).squeeze(3)
        #Conv6 = torch.nn.functional.sigmoid(self.Conv2d6(x)).squeeze(3)
        Pool2 = torch.nn.MaxPool1d(Conv2.shape[2], stride=Conv2.shape[2])
        Pool3 = torch.nn.MaxPool1d(Conv3.shape[2], stride=Conv3.shape[2])
        Pool4 = torch.nn.MaxPool1d(Conv4.shape[2], stride=Conv4.shape[2])
        #Pool5 = torch.nn.MaxPool1d(Conv5.shape[2], stride=Conv5.shape[2])
        #Pool6 = torch.nn.MaxPool1d(Conv6.shape[2], stride=Conv6.shape[2])
        torc2 = Pool2(Conv2).squeeze(2)
        torc3 = Pool3(Conv3).squeeze(2)
        torc4 = Pool4(Conv4).squeeze(2)
        #torc5 = Pool5(Conv5).squeeze(2)
        #torc6 = Pool6(Conv6).squeeze(2)
        y = torch.cat((torc2, torc3, torc4), dim=1)
        y = torch.relu(self.fc1(y))
        return self.SM(y)

def accuracy(input, std_output, model, criterion, gpu):
    if gpu:
        input = input.cuda()
        std_output = std_output.cuda()
        model = model.cuda()
    output = model(input)
    accurate = 0
    sum = 0
    for i in range(0, output.shape[0]):
        sum += 1
        one = 0
        max_p = 0
        for j in range(0, 8):
            if output[i][j] > max_p:
                max_p = output[i][j]
                one = j
        if one == std_output[i]:
            accurate += 1
    return accurate/ (sum + 0.0)
def get_loss(input, std_output, model, criterion, gpu):
    if gpu:
        input = input.cuda()
        std_output = std_output.cuda()
        model = model.cuda()
    output = model(input)
    return criterion(output, std_output)

dataloader = DataLoader(Input.word2vec_data('sina/sinanews.train'), batch_size=28, shuffle=True)
test_dataset = Input.word2vec_data('sina/sinanews.test')
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
net = CNN(input_size)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
if use_gpu:
    net = net.cuda()
optimizer = Adam(net.parameters(), lr=0.001)
loss_function = torch.nn.CrossEntropyLoss()
net.train()
epoch_num = 4000
for epoch in range(0, epoch_num):
    if epoch % 2 == 0:
        for data in test_dataloader:
            net.eval()
            print('epoch: {}/{}, accuracy: {}, loss: {}' .format(str(epoch), str(epoch_num), accuracy(data['input'], data['label2'], net, loss_function, use_gpu), get_loss(data['input'], data['label2'].squeeze().long(), net, loss_function, use_gpu)))
            net.train()
    for data in dataloader:
        x = data['input']
        #print(x.shape)
        y = data['label2'].squeeze().long()
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
        y_ = net(x)
        loss = loss_function(y_, y)
        if use_gpu:
            loss = loss.cuda()
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()