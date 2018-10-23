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
from cal_map import calculate_map, compress


# Hyper Parameters
num_epochs = 50
batch_size = 32
epoch_lr_decrease = 30
learning_rate = 0.001
encode_length = 12
num_classes = 10


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
                                 train=False,
                                 transform=test_transform)


# Construct training, query and database set
X = train_dataset.train_data
L = np.array(train_dataset.train_labels)

X = np.concatenate((X, test_dataset.test_data))
L = np.concatenate((L, np.array(test_dataset.test_labels)))

first = True

for label in range(10):
    index = np.where(L == label)[0]
    
    N = index.shape[0]
    perm = np.random.permutation(N)
    index = index[perm]
    

    data = X[index[0:100]]
    labels = L[index[0:100]]
    if first:
        test_L = labels
        test_data = data
    else:
        test_L = np.concatenate((test_L, labels))
        test_data = np.concatenate((test_data, data))

    data = X[index[100:6000]]
    labels = L[index[100:6000]]
    if first:
        dataset_L = labels
        data_set = data
    else:
        dataset_L = np.concatenate((dataset_L, labels))
        data_set = np.concatenate((data_set, data))

    data = X[index[100:600]]
    labels = L[index[100:600]]
    if first:
        train_L = labels
        train_data = data
    else:
        train_L = np.concatenate((train_L, labels))
        train_data = np.concatenate((train_data, data))

    first = False

train_dataset.train_data = train_data
train_dataset.train_labels = train_L
test_dataset.test_data = test_data
test_dataset.test_labels = test_L
database_dataset.test_data = data_set
database_dataset.test_labels = dataset_L


# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4)

database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4)

# new layer
class hash(Function):
    @staticmethod
    def forward(ctx, input):
        #ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        #input,  = ctx.saved_tensors
        #grad_output = grad_output.data

        return grad_output


def hash_layer(input):
    return hash.apply(input)


class CNN(nn.Module):
    def __init__(self, encode_length, num_classes):
        super(CNN, self).__init__()
        self.alex = torchvision.models.alexnet(pretrained=True)
        self.alex.classifier = nn.Sequential(*list(self.alex.classifier.children())[:6])
        self.fc_plus = nn.Linear(4096, encode_length)
        self.fc = nn.Linear(encode_length, num_classes, bias=False)

    def forward(self, x):
        x = self.alex.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.alex.classifier(x)
        x = self.fc_plus(x)
        code = hash_layer(x)
        # x = F.tanh(x)
        binary_out = self.fc(code)
        #binary_out = self.fc(x)

        return binary_out, x, code


cnn = CNN(encode_length=encode_length, num_classes=num_classes)
#cnn.load_state_dict(torch.load('temp.pkl'))


# Loss and Optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // epoch_lr_decrease))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
best = 0.0

# Train the Model
for epoch in range(num_epochs):
    cnn.cuda().train()
    adjust_learning_rate(optimizer, epoch)
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        binary_out, feature, _ = cnn(images)
        loss1 = criterion(binary_out, labels)
        #loss2 = F.mse_loss(torch.abs(feature), Variable(torch.ones(feature.size()).cuda()))
        loss2 = torch.mean(torch.abs(torch.pow(torch.abs(feature) - Variable(torch.ones(feature.size()).cuda()), 3)))
        loss = loss1 + 0.1 * loss2
        loss.backward()
        optimizer.step()

        if (i + 1) % (len(train_dataset) // batch_size / 2) == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f'
                   % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size,
                      loss1.data[0], loss2.data[0]))

    # Test the Model
    cnn.eval()  # Change model to 'eval' mode
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.cuda(), volatile=True)
        outputs, _, _ = cnn(images)
        _, predicted = torch.max(outputs.cpu().data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model: %.2f %%' % (100.0 * correct / total))

    if 1.0 * correct / total > best:
        best = 1.0 * correct / total
        torch.save(cnn.state_dict(), 'temp.pkl')
        
    print('best: %.2f %%' % (best * 100.0))


# Save the Trained Model
torch.save(cnn.state_dict(), 'cifar1.pkl')


# Calculate MAP
#cnn.load_state_dict(torch.load('temp.pkl'))
cnn.eval()
retrievalB, retrievalL, queryB, queryL = compress(database_loader, test_loader, cnn)
print(np.shape(retrievalB))
print(np.shape(retrievalL))
print(np.shape(queryB))
print(np.shape(queryL))

print('---calculate map---')
result = calculate_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL)
print(result)
"""
print('---calculate top map---')
result = calculate_top_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=1000)
print(result)
"""
