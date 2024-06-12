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
from python_common.services.utils import log_error


from matrice_sdk.actionTracker import ActionTracker


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
    
    global actionTracker
    actionTracker = None
    model = None
    
    try:
        actionTracker = ActionTracker(action_id)
    except Exception as e:
        log_error(__file__, 'main', f'Error initializing ActionTracker: {str(e)}')
        print(f"Error initializing ActionTracker: {str(e)}")
        sys.exit(1)
    
    stepCode='MDL_TRN_ACK'
    status='OK'
    status_description='Model Training has acknowledged'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)
    
    #model_config = actionTracker.get_job_params()
    
    actionTracker.model_config.data = f"workspace/{actionTracker.model_config['dataset_path']}/images"

    update_with_defaults(actionTracker.model_config) # Just For testing it will be removed
    print('model_config is', actionTracker.model_config)

    try:
        train_loader, val_loader, test_loader = load_data(actionTracker.model_config)
        index_to_labels = {str(idx): str(label) for idx, label in enumerate(train_loader.dataset.classes)}
        actionTracker.add_index_to_category(index_to_labels)

        model = initialize_model(actionTracker.model_config, train_loader.dataset)
        
        device = update_compute(model)
        status='OK'
        status_description='Training Dataset is loaded'

    except Exception as e:
        status='ERROR'
        status_description='Error in loading data or model' + str(e)

    stepCode='MDL_TRN_DTL'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = setup_optimizer(model, actionTracker.model_config)
    scheduler = setup_scheduler(optimizer, actionTracker.model_config)

    stepCode='MDL_TRN_STRT'
    status='OK'
    status_description='Model Training has started'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)

    best_model = None
    early_stopping=EarlyStopping(patience=actionTracker.model_config.patience,min_delta=actionTracker.model_config.min_delta)

    for epoch in range(actionTracker.model_config.epochs):

        # train for one epoch
        loss_train,acc1_train, acc5_train =train(train_loader, model, criterion, optimizer, epoch, device, actionTracker.model_config)

        # evaluate on validation set
        loss_val,acc1_val,acc5_val= validate(val_loader, model, criterion,device, actionTracker.model_config)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1_val > best_acc1
        best_acc1 = max(acc1_val, best_acc1)
        if is_best:
            best_model = model
            save_checkpoint({
                'epoch': epoch,
                'arch': actionTracker.model_config.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, model,is_best)
                    
        epochDetails= [{"splitType": "train", "metricName": "loss", "metricValue":loss_train},
                        {"splitType": "train", "metricName": "acc@1", "metricValue": acc1_train},
                        {"splitType": "train", "metricName": "acc@5", "metricValue": acc5_train},
                        {"splitType": "val", "metricName": "loss", "metricValue": loss_val},
                        {"splitType": "val", "metricName": "acc@1", "metricValue": acc1_val},
                        {"splitType": "val", "metricName": "acc@5", "metricValue": acc5_val}]

        actionTracker.log_epoch_results(epoch ,epochDetails)
        print(epoch,epochDetails)


        early_stopping.update(loss_val)
        if early_stopping.stop:
            break

    

    
    try:
        actionTracker.upload_checkpoint('model_best.pth.tar')
        actionTracker.upload_checkpoint('model_best.pt')
   
        from eval import get_metrics

        payload=[]

        if  os.path.exists(valdir):
            payload+=get_metrics('val',val_loader, best_model, index_to_labels)

        if  os.path.exists(testdir):
            payload+=get_metrics('test',test_loader, best_model, index_to_labels)
        
        actionTracker.save_evaluation_results(payload)
        status = 'SUCCESS'
        status_description='Model Training is completed'
    
    except Exception as e:
        status = 'ERROR'
        status_description = 'Model training is completed but error in model saving or eval' + str(e)
            
    stepCode='MDL_TRN_CMPL'
    
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)
    
def train(train_loader, model, criterion, optimizer, epoch, device, model_config):

    # switch to train mode
    model.train()

    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0

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
                acc5 = torch.tensor([1]).cuda(model_config.gpu).item()
            else:
                acc5 = torch.tensor([1]).item()


        total_loss += loss.item()
        total_acc1 += acc1
        total_acc5 += acc5

        # compute gradient and do step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    avg_acc1 = total_acc1 / len(train_loader)
    avg_acc5 = total_acc5 / len(train_loader)

    return avg_loss, avg_acc1, avg_acc5



def validate(val_loader, model, criterion, device, model_config):

    def run_validate(loader):
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                

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
                        acc5 = torch.tensor([1]).cuda(model_config.gpu).item()
                    else:
                        acc5 = torch.tensor([1]).item()

                total_loss += loss.item()
                total_acc1 += acc1
                total_acc5 += acc5

        avg_loss = total_loss / len(loader)
        avg_acc1 = total_acc1 / len(loader)
        avg_acc5 = total_acc5 / len(loader)

        return avg_loss, avg_acc1, avg_acc5

    model.eval()

    loss, acc1, acc5 = run_validate(val_loader)

    return loss, acc1, acc5

def load_data(model_config):
    global traindir, valdir, testdir
    traindir = os.path.join(model_config.data, 'train')
    valdir = os.path.join(model_config.data, 'val')
    testdir = os.path.join(model_config.data, 'test')
    
    
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


def initialize_model(model_config, train_dataset):
    print("=> using pre-trained model '{}'".format(model_config.arch))
    
    checkpoint_path, pretrained = actionTracker.checkpoint_path , actionTracker.pretrained
    model = models.__dict__[model_config.model_key](pretrained=pretrained)

    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load the model from the checkpoint path if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)  
        model.load_state_dict(checkpoint['state_dict'])
        print("Model loaded from checkpoint:", checkpoint_path)
    
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

def save_checkpoint(state, model,is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        if isinstance(model, torch.nn.DataParallel):
            model_best=copy.deepcopy(model.module)
        else:
            model_best=model
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
                    


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the top-k predictions for the specified values of k.

    Args:
        output (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth labels.
        topk (tuple): Tuple of integers specifying the top-k values to consider.

    Returns:
        list: List of accuracy values for each top-k value.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        acc_list = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc_list.append(correct_k.mul_(1.0 / batch_size).item())

    return acc_list



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 train.py <action_status_id>")
        sys.exit(1)
    action_status_id = sys.argv[1]
    main(action_status_id)
