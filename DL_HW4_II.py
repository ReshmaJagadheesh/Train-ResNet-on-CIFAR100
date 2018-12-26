
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
import torchvision.models as models


# In[ ]:


model = models.resnet18(pretrained=True)

model_urls = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
model.load_state_dict(model_zoo.load_url(model_urls,model_dir='./'))

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100)


# In[ ]:


transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=4)

t = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=t)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=4)


# In[ ]:


model.cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    
    total_right = 0
    total = 0
    
    for data in trainloader:
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(),Variable(labels).cuda()
        
        optimizer.zero_grad()
        
        inputs = F.interpolate(inputs,[224,224])
        
        outputs = model(inputs)
        loss = loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()
        
        predicted = outputs.data.max(1)[1]
        total += labels.size(0)
        total_right += (predicted == labels.data).float().sum()
        
    print("Training accuracy for epoch {} : {}".format(epoch+1,100*total_right/total))
    
    torch.save(model,'HW4_II_para.ckpt')
        
    my_model = torch.load('HW4_II_para.ckpt')
    
    total_right = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images,labels = data
            images, labels = Variable(images).cuda(),Variable(labels).cuda()
            
            images = F.interpolate(images,[224,224])
            
            outputs = my_model(images)
        
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            total_right += (predicted == labels.data).float().sum()
        
    print("Test accuracy: %d" % (100*total_right/total))

