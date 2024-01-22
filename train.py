import argparse
import os
import random
import shutil
import time
import warnings
import copy
import sys
from enum import Enum
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset



from python_sdk.src.actionTracker import ActionTracker
from python_sdk.matrice import Session


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# Just For testing it will be removed
def update_with_defaults(model_config):
    default_values = {
        'arch': 'resnet18',
        'workers': 4,
        'epochs': 2,
        'patience': 5,
        'min_delta': 0.5,
        'opt': 'sgd',
        'lr_scheduler': 'steplr',
        'lr_step_size': 30,
        'lr_gamma': 0.1,
        'lr_min': 0.0,
        'start_epoch': 0,
        'batch_size': 32,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'print_freq': 10,
        'resume': '',
        'evaluate': False,
        'pretrained': True,
        'world_size': -1,
        'rank': -1,
        'dist_url': 'tcp://224.66.41.62:23456',
        'dist_backend': 'nccl',
        'seed': None,
        'gpu': None,
        'multiprocessing_distributed': False,
        'dummy': False,
    }
    
    for key, value in default_values.items():
        if model_config.get(key,None) == None:
            model_config[key]= value

        if type(default_values[key]) != type(model_config[key]):
            model_config[key]=type(default_values[key])(model_config[key])
            
    return model_config

best_acc1 = 0


def main(action_id):
    global best_acc1
    session=Session()
    actionTracker = ActionTracker(session,action_id)
    
    stepCode='MDL_TRN_ACK'
    status='OK'
    status_description='Model Training has acknowledged'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)
    
    model_config = actionTracker.get_job_params()
    
    _idDataset = model_config['_idDataset']
    dataset_version = model_config['dataset_version']
    model_config.data = f'workspace/{str(_idDataset)}-{str(dataset_version).lower()}-imagenet/images'
    print(f"epochs is :{model_config.keys()}")
    update_with_defaults(model_config) # Just For testing it will be removed
    print('model_config is', model_config)
    print(f"epochs is :{model_config.epochs}")
    train_loader, val_loader, test_loader = load_data(model_config)
    index_to_labels = {str(idx): str(label) for idx, label in enumerate(train_loader.dataset.classes)}
    actionTracker.add_index_to_category(index_to_labels)

    model = initialize_model(model_config, train_loader.dataset)
    
    device = update_compute(model)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = setup_optimizer(model, model_config)

    scheduler = setup_scheduler(optimizer, model_config)




    stepCode='MDL_TRN_DTL'
    status='OK'
    status_description='Training Dataset is loaded'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)


    stepCode='MDL_TRN_STRT'
    status='OK'
    status_description='Model Training has started'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)


    early_stopping=EarlyStopping(patience=model_config.patience,min_delta=model_config.min_delta)

    for epoch in range(model_config.epochs):

        # train for one epoch
        loss_train,acc1_train, acc5_train =train(train_loader, model, criterion, optimizer, epoch, device, model_config)

        # evaluate on validation set
        loss_val,acc1_val,acc5_val,acc1= validate(val_loader, model, criterion,device, model_config)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)


        save_checkpoint({
            'epoch': epoch + 1,
            'arch': model_config.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best)
                


        epochDetails= [{"splitType": "train", "metricName": "loss", "metricValue":loss_train},
                        {"splitType": "train", "metricName": "acc@1", "metricValue": acc1_train},
                        {"splitType": "train", "metricName": "acc@5", "metricValue": acc5_train},
                        {"splitType": "val", "metricName": "loss", "metricValue": loss_val},
                        {"splitType": "val", "metricName": "acc@1", "metricValue": acc1_val},
                        {"splitType": "val", "metricName": "acc@5", "metricValue": acc5_val}]

        actionTracker.log_epoch_results(epoch + 1,epochDetails)
        print(epoch,epochDetails)


        # early_stopping.update(loss_val)
        # if early_stopping.stop:
        #    break

    stepCode='MDL_TRN_CMPL'
    status='SUCCESS'
    status_description='Model Training is completed'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)

    
    try:
        actionTracker.upload_checkpoint('model_best.pth.tar')
        actionTracker.upload_checkpoint('model_best.pt')
    except:
        print("Couldn't upload model_best.pt")

    from eval import get_metrics

    payload=[]
    
    if os.path.exists(traindir):
        payload+=get_metrics('train',train_loader, model,index_to_labels)

    if  os.path.exists(valdir):
        payload+=get_metrics('val',val_loader, model,index_to_labels)

    if  os.path.exists(testdir):
        payload+=get_metrics('test',test_loader, model,index_to_labels)
    
    actionTracker.save_evaluation_results(payload)
    
def train(train_loader, model, criterion, optimizer, epoch, device, model_config):

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        # move data to the same device as model
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))[0]

        try:
            acc5 = accuracy(output, target, topk=(5,))[0]
        except:
            if torch.cuda.is_available():
                acc5 = torch.tensor([100]).cuda(model_config.gpu)
            else:
                acc5 = torch.tensor([100])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()

    return loss.item(),acc1.item(), acc5.item()



def validate(val_loader, model, criterion, device,model_config):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                

                images = images.to(device)
                target = target.to(device)


                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1 = accuracy(output, target, topk=(1,))[0]

                try:
                    acc5 = accuracy(output, target, topk=(5,))[0]
                except:
                    if torch.cuda.is_available():
                        acc5 = torch.tensor([100]).cuda(model_config.gpu)
                    else:
                        acc5 = torch.tensor([100])
                        
                top1.update(acc1[0], images.size(0))

            return loss.item(),acc1.item(), acc5.item(),top1

    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)

    model.eval()

    loss,acc1, acc5, top1=run_validate(val_loader)


    return loss,acc1, acc5, top1.avg

def load_data(model_config):
    global traindir, valdir, testdir
    traindir = os.path.join(model_config.data, 'train')
    valdir = os.path.join(model_config.data, 'val')
    testdir = os.path.join(model_config.data, 'test')
    
    try:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=model_config.batch_size, shuffle=False,
            num_workers=model_config.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=model_config.batch_size, shuffle=False,
            num_workers=model_config.workers, pin_memory=True)

        test_loader = None
        if os.path.exists(testdir):
            test_dataset = datasets.ImageFolder(
                testdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
            
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=model_config.batch_size, shuffle=False,
                num_workers=model_config.workers, pin_memory=True)

        return train_loader, val_loader, test_loader

    except Exception as e:
        print(f"=> Dummy data is used! Error: {e}")
        train_dataset = datasets.FakeData(150, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(30, (3, 224, 224), 1000, transforms.ToTensor())
        test_dataset = datasets.FakeData(30, (3, 224, 224), 1000, transforms.ToTensor())
        return torch.utils.data.DataLoader(train_dataset), torch.utils.data.DataLoader(val_dataset), torch.utils.data.DataLoader(test_dataset)

def initialize_model(model_config, train_dataset):
    print("=> using pre-trained model '{}'".format(model_config.arch))
    model = models.__dict__[model_config.arch](pretrained=True)

    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def setup_optimizer(model, model_config):
    opt_name = model_config.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=model_config.lr,
            momentum=model_config.momentum,
            weight_decay=model_config.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=model_config.lr, momentum=model_config.momentum, weight_decay=model_config.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.lr, weight_decay=model_config.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {model_config.opt}. Only SGD, RMSprop and AdamW are supported.")

    return optimizer

def setup_scheduler(optimizer, model_config):
    model_config.lr_scheduler = model_config.lr_scheduler.lower()
    if model_config.lr_scheduler == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=model_config.lr_step_size, gamma=model_config.lr_gamma)
    elif model_config.lr_scheduler == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=model_config.epochs, eta_min=model_config.lr_min
        )
    elif model_config.lr_scheduler == "exponentiallr":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=model_config.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{model_config.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    return scheduler

def update_compute(model):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        print('using CPU, this will be slow')
        device = torch.device("cpu")

    model = model.to(device)
    
    return device

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        if isinstance(state, torch.nn.DataParallel):
            model_best=copy.deepcopy(state.module)
        else:
            model_best=state
        torch.save(model_best,'model_best.pt')

class EarlyStopping:
    def __init__(self, patience=5,min_delta=10):
        self.patience = patience
        self.counter = 0
        self.lowest_loss = None
        self.min_delta = min_delta
        self.stop = False

    def update(self, val_loss):
        if self.lowest_loss is None:
            self.lowest_loss = val_loss

        elif self.lowest_loss-val_loss  > self.min_delta:
            self.lowest_loss = val_loss
            self.counter = 0

        elif self.lowest_loss-val_loss  < self.min_delta:
            self.counter += 1
            print(f'Early stoping count is {self.counter}')
            if self.counter >= self.patience:
                self.stop = True
                    
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 train.py <action_status_id>")
        sys.exit(1)
    action_status_id = sys.argv[1]
    main(action_status_id)