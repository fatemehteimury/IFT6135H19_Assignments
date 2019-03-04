from comet_ml import Experiment, ExistingExperiment

import sys
import os
import time
import shutil
import random
import argparse
from os.path import realpath, dirname, split, basename, join

import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
cudnn.benchmark = True

hyperparameters = {
        'type': 'ClassicNet',
        'filters': [64, 128, 256, 512, 1024],
        'layers': [1, 2, 2, 4, 4],
        'colorjitter': {
            'brightness': 0.4,
            'contrast': 0.4,
            'saturation': 0.4,
            'hue': 0.2
            },
        'randomaffine': {
            'degrees': 12,
            'shear': 8
            },
        'randomresizedcrop': {
            'size': 64,
            'scale': (0.4, 1.0),
            'ratio': (1, 1)
            }
        }

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--num-workers', type=int, default=12)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--learning-rate', type=float, default=1e-1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=36789)
parser.add_argument('--evaluate', action='store_true', default=False) # TODO
parser.add_argument('--api-key', type=str, default=None)
args = parser.parse_args()

random.seed(args.seed)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

class ClassicUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ClassicUnit, self).__init__()
        self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ResidualUnit, self).__init__()
        self.layer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, mid_channels, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, 3, stride=1, padding=1))
    
    def forward(self, x):
        o = self.layer(x)
        r = F.pad(x, (0,0,0,0,0,o.size()[1]-x.size()[1]))
        return o + r

class ResidualBottleneckUnit(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ResidualBottleneckUnit, self).__init__()
        self.layer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, mid_channels, 1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, 1, stride=1, padding=0))

    def forward(self, x):
        o = self.layer(x)
        r = F.pad(x, (0,0,0,0,0,o.size()[1]-x.size()[1]))
        return o + r

class ClassicBlock(nn.Module):
    def __init__(self, ic, oc, nu):
        super(ClassicBlock, self).__init__()
        assert (oc-ic) % nu == 0
        l = []
        s = int((oc-ic)/nu)
        for i in range(ic, oc, s):
            l.append(ClassicUnit(i, i+s))
        self.layer = nn.Sequential(*l)

    def forward(self, x):
        return self.layer(x)

class ResidualBlock(nn.Module):
    def __init__(self, ic, oc, nu):
        super(ResidualBlock, self).__init__()
        assert (oc-ic) % nu == 0
        assert ((oc-ic)/nu) % 2 == 0
        l = []
        s = int((oc-ic)/nu)
        for i in range(ic, oc, s):
            l.append(ResidualUnit(i, i+int(s/2), i+s))
        self.layer = nn.Sequential(*l)

    def forward(self, x):
        return self.layer(x)

class ResidualBottleneckBlock(nn.Module):
    def __init__(self, ic, oc, fr, nu):
        super(ResidualBottleneckBlock, self).__init__()
        assert (oc-ic) % nu == 0
        assert ic % fr == 0 and ((oc-ic)/nu) % fr == 0
        l = []
        s = int((oc-ic)/nu)
        for i in range(ic, oc, s):
            l.append(ResidualBottleneckUnit(i, int(i/fr), i+s))
        self.layer = nn.Sequential(*l)

    def forward(self, x):
        return self.layer(x)

class ClassicNet(nn.Module):
    def __init__(self, filters, layers):
        super(ClassicNet, self).__init__()
        self.conv0 = ClassicBlock(         3, filters[0], layers[0])
        self.conv1 = ClassicBlock(filters[0], filters[1], layers[1])
        self.conv2 = ClassicBlock(filters[1], filters[2], layers[2])
        self.conv3 = ClassicBlock(filters[2], filters[3], layers[3])
        self.conv4 = ClassicBlock(filters[3], filters[4], layers[4])
        self.linear = nn.Linear(filters[4], 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(8, 8)

    def forward(self, x):
        x = self.conv0(x)
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.maxpool(self.conv3(x))
        x = self.avgpool(self.conv4(x))
        x = torch.squeeze(x)
        x = self.linear(x)
        return x

class ResNet(nn.Module):
    def __init__(self, filters, layers):
        super(ResNet, self).__init__()
        self.conv0 = nn.Conv2d(             3, filters[0], 3, stride=1, padding=1)
        self.conv1 = ResidualBlock(filters[0], filters[1], layers[0])
        self.conv2 = ResidualBlock(filters[1], filters[2], layers[1])
        self.conv3 = ResidualBlock(filters[2], filters[3], layers[2])
        self.conv4 = nn.Sequential(
                ResidualBlock(     filters[3], filters[4], layers[3]),
                nn.ReLU(inplace=True))
        self.linear = nn.Linear(filters[4], 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(8, 8)

    def forward(self, x):
        x = self.conv0(x)
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.maxpool(self.conv3(x))
        x = self.avgpool(self.conv4(x))
        x = torch.squeeze(x)
        x = self.linear(x)
        return x

class ResNetBottleneck(nn.Module):
    def __init__(self, filters, layers, bottleneck):
        super(ResNetBottleneck, self).__init__()
        self.conv0 = nn.Conv2d(                       3, filters[0], 3, stride=1, padding=1)
        self.conv1 = ResidualBottleneckBlock(filters[0], filters[1], bottleneck[0], layers[0])
        self.conv2 = ResidualBottleneckBlock(filters[1], filters[2], bottleneck[1], layers[1])
        self.conv3 = ResidualBottleneckBlock(filters[2], filters[3], bottleneck[2], layers[2])
        self.conv4 = nn.Sequential(
                ResidualBottleneckBlock(     filters[3], filters[4], bottleneck[3], layers[3]),
                nn.ReLU(inplace=True))
        self.linear = nn.Linear(filters[4], 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(8, 8)

    def forward(self, x):
        x = self.conv0(x)
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.maxpool(self.conv3(x))
        x = self.avgpool(self.conv4(x))
        x = torch.squeeze(x)
        x = self.linear(x)
        return x

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.len = len(dataset)
        self.transforms = transforms

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transforms:
            image = self.transforms(image)
        label = torch.FloatTensor([label])
        return image, label

    def __len__(self):
        return self.len

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dir='testset/test', ext='.jpg', transforms=None):
        self.img_ids = sorted([int(i[:-len(ext)]) for i in os.listdir(dir)])
        self.dir = dir
        self.ext = ext
        self.transforms = transforms
        self.len = len(self.img_ids)

    def load_img(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        path = join(self.dir, str(img_id) + self.ext)
        img = self.load_img(path)
        if self.transforms:
            img = self.transforms(img)
        return img, img_id

    def __len__(self):
        return self.len

class EarlyStopping(object):
    def __init__(self, mode='min', early_stopping_threshold=0.0001, early_stopping_patience=28):
        self.mode = mode
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_epoch = 0
        self.early_stopping_score = None
        self.current_epoch = 0

    def step(self, score):
        self.current_epoch += 1
        if (self.early_stopping_score is None or 
                (self.mode == 'max' and score > (1+self.early_stopping_threshold)*self.early_stopping_score) or 
                (self.mode == 'min' and score < (1-self.early_stopping_threshold)*self.early_stopping_score)):
            self.early_stopping_score = score
            self.early_stopping_epoch = self.current_epoch
        if self.current_epoch - self.early_stopping_epoch > self.early_stopping_patience:
            raise Exception

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

def train(train_dataloader, validation_dataloader, mnum, experiment):
    if 'type' not in hyperparameters:
        raise Exception
    if hyperparameters['type'] == 'ClassicNet':
        model = ClassicNet(hyperparameters['filters'], hyperparameters['layers'])
    elif hyperparameters['type'] == 'ResNet':
        model = ResNet(hyperparameters['filters'], hyperparameters['layers'])
    elif hyperparameters['type'] == 'ResNetBottleneck':
        model = ResNetBottleneck(hyperparameters['filters'], hyperparameters['layers'], hyperparameters['bottleneck'])
    else:
        raise Exception
    model.apply(init_weights)
    model = model.cuda()
    
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    vx = {p:torch.zeros_like(p.data) for p in model.parameters()}
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    early_stopping = EarlyStopping()
    best_acc = 0
    for epoch in range(0, args.epochs):
        running_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
    
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            for p in model.parameters():
                vx[p].mul_(args.momentum).add_(p.grad.data)
                p.grad.data = vx[p]
            optimizer.step()
    
            running_loss += loss.item() * labels.size(0)
            predicted = torch.round(torch.sigmoid(outputs.data))
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        loss = running_loss / total
        acc = 100 * float(correct) / total
        if experiment:
            with experiment.train():
                experiment.log_metric('loss'+str(mnum), loss, step=epoch)
                experiment.log_metric('acc'+str(mnum), acc, step=epoch)
        print('TRAIN - Epoch: %d, Loss%d: %.3f, Accuracy%d: %.3f' % (
            epoch, mnum, loss, mnum, acc))
        
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in validation_dataloader:
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
    
                running_loss += loss.item() * labels.size(0)
                predicted = torch.round(torch.sigmoid(outputs.data))
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        loss = running_loss / total
        acc = 100 * float(correct) / total
        if experiment:
            with experiment.validate():
                experiment.log_metric('loss'+str(mnum), loss, step=epoch)
                experiment.log_metric('acc'+str(mnum), acc, step=epoch)
        print('VALIDATION - Epoch: %d, Loss%d: %.3f, Accuracy%d: %.3f' % (
            epoch, mnum, loss, mnum, acc))
        print('Epoch Time: %.1f' % (time.time()-running_time))
        print()
        
        scheduler.step(loss)
        early_stopping.step(loss)
        
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        checkpoint = {
            'state_dict': model.state_dict()
            }
        if experiment:
            checkpoint['experiment_key'] = experiment.get_key()
        save_checkpoint(checkpoint, is_best, filename='checkpoint'+str(mnum)+'.pth.tar', best_filename='model_best'+str(mnum)+'.pth.tar')
    print('Finished Training Model ' + str(mnum))

def evaluate(dataloader, model1, model2):
    # Mapping: {'Cat': 0, 'Dog': 1}
    with open('submission.csv', 'w+') as f:
        f.write('id,label\n')
        for imgs, img_ids in dataloader:
            imgs = imgs.cuda()
            outputs = model1(imgs) + model2(imgs)
            predicted = torch.round(torch.sigmoid(outputs.data))
            for i in range(predicted.size(0)):
                f.write(str(int(img_ids[i].item())) + ',' + ['Cat','Dog'][int(predicted[i].item())] + '\n')
    print('Done Generating Submission... Exiting...')
    sys.exit()

if args.evaluate:
    print('Evaluating on Test Set...')
    try:
        dataset = TestDataset(transforms=transforms.ToTensor())
    except:
        import zipfile
        zip_ref = zipfile.ZipFile('testset.zip', 'r')
        zip_ref.extractall()
        zip_ref.close()
        dataset = TestDataset(transforms=transforms.ToTensor())

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    checkpoint1 = torch.load('model_best1.pth.tar')
    model1 = ClassicNet(hyperparameters['filters'], hyperparameters['layers'])
    model1 = model1.cuda()
    model1.load_state_dict(checkpoint1['state_dict'])

    checkpoint2 = torch.load('model_best2.pth.tar')
    model2 = ClassicNet(hyperparameters['filters'], hyperparameters['layers'])
    model2 = model2.cuda()
    model2.load_state_dict(checkpoint2['state_dict'])
    
    evaluate(dataloader, model1, model2)

experiment = None
if args.api_key:
    project_dir, experiment_name = split(dirname(realpath(__file__)))
    project_name = basename(project_dir)
    experiment = Experiment(
        api_key=args.api_key,
        project_name=project_name,
        auto_param_logging=False,
        auto_metric_logging=False,
        parse_args=False)
    experiment.log_other('experiment_name', experiment_name)
    experiment.log_parameters(vars(args))
    for k in hyperparameters:
        if type(hyperparameters[k]) == dict:
            experiment.log_parameters(hyperparameters[k], prefix=k)
        else:
            experiment.log_parameter(k, hyperparameters[k])

try:
    dataset = torchvision.datasets.ImageFolder(root='./trainset')
except:
    import zipfile
    zip_ref = zipfile.ZipFile('trainset.zip', 'r')
    zip_ref.extractall()
    zip_ref.close()
    dataset = torchvision.datasets.ImageFolder(root='./trainset')
print(dataset.class_to_idx)

offset_m1, offset_m2 = int(0.20*len(dataset)), int(0.80*len(dataset))
indices = np.random.permutation(len(dataset))
train_indices_m1, validation_indices_m1 = indices[offset_m1:], indices[:offset_m1]
train_indices_m2, validation_indices_m2 = indices[:offset_m2], indices[offset_m2:]
train_dataset_m1, validation_dataset_m1 = torch.utils.data.Subset(dataset, train_indices_m1), torch.utils.data.Subset(dataset, validation_indices_m1)
train_dataset_m2, validation_dataset_m2 = torch.utils.data.Subset(dataset, train_indices_m2), torch.utils.data.Subset(dataset, validation_indices_m2)

train_transform = transforms.Compose([
    transforms.ColorJitter(
        brightness=hyperparameters['colorjitter']['brightness'],
        contrast=hyperparameters['colorjitter']['contrast'], 
        saturation=hyperparameters['colorjitter']['saturation'], 
        hue=hyperparameters['colorjitter']['hue']),
    transforms.RandomAffine(
        hyperparameters['randomaffine']['degrees'], 
        shear=hyperparameters['randomaffine']['shear']),
    transforms.RandomResizedCrop(
        hyperparameters['randomresizedcrop']['size'],
        scale=hyperparameters['randomresizedcrop']['scale'], 
        ratio=hyperparameters['randomresizedcrop']['ratio']),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

train_dataset_m1, validation_dataset_m1 = ImageDataset(train_dataset_m1, transforms=train_transform), ImageDataset(validation_dataset_m1, transforms=transforms.ToTensor())
train_dataloader_m1 = torch.utils.data.DataLoader(train_dataset_m1, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
validation_dataloader_m1 = torch.utils.data.DataLoader(validation_dataset_m1, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

train_dataset_m2, validation_dataset_m2 = ImageDataset(train_dataset_m2, transforms=train_transform), ImageDataset(validation_dataset_m2, transforms=transforms.ToTensor())
train_dataloader_m2 = torch.utils.data.DataLoader(train_dataset_m2, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
validation_dataloader_m2 = torch.utils.data.DataLoader(validation_dataset_m2, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

print('Train Model 1')
try:
    train(train_dataloader_m1, validation_dataloader_m1, 1, experiment)
except:
    pass

print('Train Model 2')
try:
    train(train_dataloader_m2, validation_dataloader_m2, 2, experiment)
except:
    pass
