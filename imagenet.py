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
from PIL import Image
import os
import os.path
import matplotlib.image as mpimg
from cal_map import calculate_top_map, compress


# Hyper Parameters
num_epochs = 100
batch_size = 32
epoch_lr_decrease = 80
learning_rate = 0.001
encode_length = 16
num_classes = 100


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class IMAGENET(torch.utils.data.Dataset):

    def __init__(self, root,
                 transform=None, target_transform=None, train=True, database_bool=False):
        self.loader = default_loader
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.base_folder = 'train.txt'
        elif database_bool:
            self.base_folder = 'database.txt'
        else:
            self.base_folder = 'test.txt'

        self.train_data = []
        self.train_labels = []

        filename = os.path.join(self.root, self.base_folder)
        # fo = open(file, 'rb')

        with open(filename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                # print lines.split()
                if not lines:
                    break
                pos_tmp = lines.split()[0]
                # print pos_tmp
                pos_tmp = os.path.join(self.root, pos_tmp)
                label_tmp = lines.split()[1:]
                self.train_data.append(pos_tmp)
                self.train_labels.append(label_tmp)
        self.train_data = np.array(self.train_data)
        # self.train_labels.reshape()
        self.train_labels = np.array(self.train_labels, dtype=np.float)
        self.train_labels.reshape((-1, num_classes))

        # self.train_data = np.concatenate(self.train_data)
        # self.train_data = self.train_data.reshape((50000, 3, 32, 32))
        # self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.train_data[index], self.train_labels[index]
        target = int(np.where(target == 1)[0])

        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.train_data)


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Dataset
train_dataset = IMAGENET(root='data/imagenet',
                         train=True,
                         transform=train_transform)

test_dataset = IMAGENET(root='data/imagenet',
                        train=False,
                        transform=test_transform)

database_dataset = IMAGENET(root='data/imagenet',
                           train=False,
                           transform=test_transform,
                           database_bool=True)


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
        output = self.fc(code)

        return output, x, code


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
        outputs, feature, _ = cnn(images)
        loss1 = criterion(outputs, labels)
        #loss2 = F.mse_loss(torch.abs(feature), Variable(torch.ones(feature.size()).cuda()))
        loss2 = torch.mean(torch.abs(torch.pow(torch.abs(feature) - Variable(torch.ones(feature.size()).cuda()), 3)))
        loss = loss1 + 1 * loss2
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
torch.save(cnn.state_dict(), 'imagenet.pkl')


# Calculate MAP
#cnn.load_state_dict(torch.load('temp.pkl'))
cnn.eval()
retrievalB, retrievalL, queryB, queryL = compress(database_loader, test_loader, cnn, classes=num_classes)
print(np.shape(retrievalB))
print(np.shape(retrievalL))
print(np.shape(queryB))
print(np.shape(queryL))
"""
print('---calculate map---')
result = calculate_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL)
print(result)
"""
print('---calculate top map---')
result = calculate_top_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=1000)
print(result)

