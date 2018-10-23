import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import math
import numpy as np
from cal_map import calculate_top_map, compress


# Hyper Parameters
num_epochs = 60
batch_size = 32
# epoch_lr_decrease = 300
learning_rate = 0.0001
encode_length = 64

if encode_length == 16:
    num_epochs = 300


train_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset
train_dataset = dsets.CIFAR10(root='data/',
                              train=True,
                              transform=train_transform,
                              download=True)

test_dataset = dsets.CIFAR10(root='data/',
                             train=False,
                             transform=test_transform)

database_dataset = dsets.CIFAR10(root='data/',
                                 train=True,
                                 transform=test_transform)

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=4)

database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)


# new layer
class hash(Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input,  = ctx.saved_tensors
        # grad_output = grad_output.data

        return grad_output


def hash_layer(input):
    return hash.apply(input)


class CNN(nn.Module):
    def __init__(self, encode_length):
        super(CNN, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.fc_encode = nn.Linear(4096, encode_length)


    def forward(self, x):
        x = self.vgg.features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)
        h = self.fc_encode(x)
        b = hash_layer(h)

        return x, h, b


cnn = CNN(encode_length=encode_length)
#cnn.load_state_dict(torch.load('vgg.pkl'))


optimizer = torch.optim.SGD(cnn.fc_encode.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

"""
def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // epoch_lr_decrease))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
"""
        
best = 0.0

# Train the Model
for epoch in range(num_epochs):
    cnn.cuda().train()
    # adjust_learning_rate(optimizer, epoch)
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        x, h, b = cnn(images)

        target_b = F.cosine_similarity(b[:labels.size(0) / 2], b[labels.size(0) / 2:])
        target_x = F.cosine_similarity(x[:labels.size(0) / 2], x[labels.size(0) / 2:])
        loss1 = F.mse_loss(target_b, target_x)
        #loss2 = F.mse_loss(torch.abs(h), Variable(torch.ones(h.size()).cuda()))
        loss2 = torch.mean(torch.abs(torch.pow(torch.abs(h) - Variable(torch.ones(h.size()).cuda()), 3)))
        loss = loss1 + 0.1 * loss2
        loss.backward()
        optimizer.step()

        if (i + 1) % (len(train_dataset) // batch_size / 1) == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f'
                   % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size,
                      loss1.data[0], loss2.data[0]))
    
    # Save the Trained Model
    torch.save(cnn.state_dict(), 'vgg.pkl')

    # Test the Model
    if (epoch + 1) % 5 == 0:
        cnn.eval()
        retrievalB, retrievalL, queryB, queryL = compress(train_loader, test_loader, cnn)
        # print(np.shape(retrievalB))
        # print(np.shape(retrievalL))
        # print(np.shape(queryB))
        # print(np.shape(queryL))
        """
        print('---calculate map---')
        result = calculate_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL)
        print(result)
        """
        print('---calculate top map---')
        result = calculate_top_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=1000)
        print(result)

        if result > best:
            best = result
            torch.save(cnn.state_dict(), 'temp.pkl')
        
        print('best: %.6f' % (best))

