
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms


# In[2]:


transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=4)

t = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=t)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=4)


# In[16]:


class ResNet(nn.Module):
    def __init__(self,BasicBlock):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,stride=1,padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout2d(p=0.2)
        self.BasicBlock1 = BasicBlock(32,32,1)
        self.BasicBlock2 = BasicBlock(32,32,1)
        self.BasicBlock3 = BasicBlock(32,64,2)
        self.BasicBlock4 = BasicBlock(64,64,1)
        self.BasicBlock5 = BasicBlock(64,64,1)
        self.BasicBlock6 = BasicBlock(64,64,1)
        self.BasicBlock8 = BasicBlock(64,128,2)
        self.BasicBlock9 = BasicBlock(128,128,1)
        self.BasicBlock10 = BasicBlock(128,128,1)
        self.BasicBlock11 = BasicBlock(128,128,1)
        self.BasicBlock12 = BasicBlock(128,256,2)
        self.BasicBlock13 = BasicBlock(256,256,1)
        self.fc1 = nn.Linear(1024,100)
        
    def forward(self,x):
        x = self.dropout(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.BasicBlock1(x)
        x = self.BasicBlock2(x)
        x = self.BasicBlock3(x)
        x = self.BasicBlock4(x)
        x = self.BasicBlock5(x)
        x = self.BasicBlock6(x)
        x = self.BasicBlock8(x)
        x = self.BasicBlock9(x)
        x = self.BasicBlock10(x)
        x = self.BasicBlock11(x)
        x = self.BasicBlock12(x)
        x = F.max_pool2d(self.BasicBlock13(x),3,stride=2,padding=1)
        x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        
        return x
        
class BasicBlock(nn.Module):
    def __init__(self,in_planes,out_planes,s):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_planes,out_planes,3,stride=s,padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes,out_planes,3,stride=s,padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_planes)
        self.downsample = nn.Sequential()
        if (s!=1):
            self.downsample = nn.Sequential(nn.Conv2d(in_planes,out_planes,1,stride=2),nn.BatchNorm2d(out_planes))
        
    def forward(self,x):
        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = self.conv2_bn(self.conv2(out))
        x = self.downsample(x)
        out = F.interpolate(out,[x.shape[2],x.shape[3]])
        out += x
        return out
            


# In[ ]:


model = ResNet(BasicBlock)
model.cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

num_epochs = 70

for epoch in range(num_epochs):
    
    total_right = 0
    total = 0
    
    for data in trainloader:
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(),Variable(labels).cuda()
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()
        
        predicted = outputs.data.max(1)[1]
        total += labels.size(0)
        total_right += (predicted == labels.data).float().sum()
        
    print("Training accuracy for epoch {} : {}".format(epoch+1,total_right/total))
    
    if (epoch+1)%5==0:
        torch.save(model,'hw4_para.ckpt')
        
    if (epoch+1)%10==0:
        my_model = torch.load('hw4_para.ckpt')

        total_right = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                images,labels = data
                images, labels = Variable(images).cuda(),Variable(labels).cuda()
                outputs = my_model(images)
        
                predicted = outputs.data.max(1)[1]
                total += labels.size(0)
                total_right += (predicted == labels.data).float().sum()
        
        print("Test accuracy: %d" % (100*total_right/total))

