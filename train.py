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
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

from matrice_sdk.src.actionTracker import ActionTracker
from matrice_sdk.src.matrice import Session

# List of available model names in torchvision.models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Default configuration
DEFAULT_CONFIG = {
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

best_acc1 = 0

def update_with_defaults(config):
    """
    Update the provided config with default values from DEFAULT_CONFIG,
    ensuring correct types for each parameter.
    """
    for key, value in DEFAULT_CONFIG.items():
        config[key] = type(value)(config.get(key, value))
    return config

def main(action_id):
    """
    Main function to start model training based on the given action_id.
    It handles data loading, model initialization, training, validation,
    and evaluation.
    """
    global best_acc1
    session = Session()
    actionTracker = ActionTracker(session, action_id)
    
    actionTracker.update_status('MDL_TRN_ACK', 'OK', 'Model Training has acknowledged')
    
    # Get model configuration from action tracker and update with defaults
    model_config = actionTracker.get_job_params()
    model_config.data = f'workspace/{model_config["_idDataset"]}-{model_config["dataset_version"].lower()}-imagenet/images'
    update_with_defaults(model_config)
    print('model_config is', model_config)

    try:
        # Load data and initialize model
        train_loader, val_loader, test_loader = load_data(model_config)
        index_to_labels = {str(idx): str(label) for idx, label in enumerate(train_loader.dataset.classes)}
        actionTracker.add_index_to_category(index_to_labels)

        model = initialize_model(model_config, train_loader.dataset)
        device = update_compute(model)
        actionTracker.update_status('MDL_TRN_DTL', 'OK', 'Training Dataset is loaded')
    except Exception as e:
        actionTracker.update_status('MDL_TRN_DTL', 'ERROR', 'Error in loading data or model' + str(e))
        return

    # Setup loss criterion, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = setup_optimizer(model, model_config)
    scheduler = setup_scheduler(optimizer, model_config)

    actionTracker.update_status('MDL_TRN_STRT', 'OK', 'Model Training has started')

    best_model = None
    early_stopping = EarlyStopping(patience=model_config.patience, min_delta=model_config.min_delta)

    for epoch in range(model_config.epochs):
        # Training and validation for each epoch
        loss_train, acc1_train, acc5_train = train(train_loader, model, criterion, optimizer, epoch, device, model_config)
        loss_val, acc1_val, acc5_val = validate(val_loader, model, criterion, device, model_config)

        scheduler.step()
        is_best = acc1_val > best_acc1
        best_acc1 = max(acc1_val, best_acc1)
        if is_best:
            best_model = model
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': model_config.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, model, is_best)

        # Log epoch results to action tracker
        epochDetails = [
            {"splitType": "train", "metricName": "loss", "metricValue": loss_train},
            {"splitType": "train", "metricName": "acc@1", "metricValue": acc1_train},
            {"splitType": "train", "metricName": "acc@5", "metricValue": acc5_train},
            {"splitType": "val", "metricName": "loss", "metricValue": loss_val},
            {"splitType": "val", "metricName": "acc@1", "metricValue": acc1_val},
            {"splitType": "val", "metricName": "acc@5", "metricValue": acc5_val}
        ]

        actionTracker.log_epoch_results(epoch + 1, epochDetails)
        print(epoch, epochDetails)

        early_stopping.update(loss_val)
        if early_stopping.stop:
            break

    try:
        # Upload best model checkpoint and evaluate
        actionTracker.upload_checkpoint('model_best.pth.tar')
        actionTracker.upload_checkpoint('model_best.pt')

        from eval import get_metrics

        payload = []
        if os.path.exists(valdir):
            payload += get_metrics('val', val_loader, best_model, index_to_labels)
        if os.path.exists(testdir):
            payload += get_metrics('test', test_loader, best_model, index_to_labels)

        actionTracker.save_evaluation_results(payload)
        actionTracker.update_status('MDL_TRN_CMPL', 'SUCCESS', 'Model Training is completed')
    except Exception as e:
        actionTracker.update_status('MDL_TRN_CMPL', 'ERROR', 'Model training completed but error in model saving or eval: ' + str(e))

def train(loader, model, criterion, optimizer, epoch, device, config):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_acc1, total_acc5 = 0.0, 0.0, 0.0

    for images, target in loader:
        images, target = images.to(device), target.to(device)
        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        total_loss += loss.item()
        total_acc1 += acc1
        total_acc5 += acc5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(loader), total_acc1 / len(loader), total_acc5 / len(loader)

def validate(loader, model, criterion, device, config):
    """
    Validate the model on the validation set.
    """
    model.eval()
    total_loss, total_acc1, total_acc5 = 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            total_loss += loss.item()
            total_acc1 += acc1
            total_acc5 += acc5

    return total_loss / len(loader), total_acc1 / len(loader), total_acc5 / len(loader)

def load_data(config):
    """
    Load training, validation, and test datasets.
    """
    global traindir, valdir, testdir
    traindir = os.path.join(config.data, 'train')
    valdir = os.path.join(config.data, 'val')
    testdir = os.path.join(config.data, 'test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Training data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True)

    # Validation data loader
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True)

    # Test data loader
    test_loader = None
    if os.path.exists(testdir):
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(testdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def initialize_model(config, train_dataset):
    """
    Initialize the model architecture and modify the final layer
    for the specific dataset.
    """
    print(f"=> using pre-trained model '{config.arch}'")
    model = models.__dict__[config.arch](pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    return model

def setup_optimizer(model, config):
    """
    Setup optimizer based on configuration.
    """
    opt_name = config.opt.lower()
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    elif opt_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Invalid optimizer {config.opt}")

def setup_scheduler(optimizer, config):
    """
    Setup learning rate scheduler based on configuration.
    """
    if config.lr_scheduler == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
    elif config.lr_scheduler == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr_min)
    elif config.lr_scheduler == "exponentiallr":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)
    else:
        raise ValueError(f"Invalid lr scheduler {config.lr_scheduler}")

def update_compute(model):
    """
    Move model to appropriate computing device (GPU, MPS, or CPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return device

def save_checkpoint(state, model, is_best, filename='checkpoint.pth.tar'):
    """
    Save model checkpoint and best model separately.
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        torch.save(copy.deepcopy(model.module if isinstance(model, nn.DataParallel) else model), 'model_best.pt')

class EarlyStopping:
    """
    Early stopping to terminate training when validation loss does not improve.
    """
    def __init__(self, patience=5, min_delta=10):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.lowest_loss = None
        self.stop = False

    def update(self, val_loss):
        if self.lowest_loss is None or (self.lowest_loss - val_loss) > self.min_delta:
            self.lowest_loss = val_loss
            self.counter = 0
        elif (self.lowest_loss - val_loss) < self.min_delta:
            self.counter += 1
            print(f'Early stopping count is {self.counter}')
            if self.counter >= self.patience:
                self.stop = True

def accuracy(output, target, topk=(1,)):
    """
    Compute accuracy at top-k for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        correct = pred.t().eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(1.0 / batch_size).item() for k in topk]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 train.py <action_status_id>")
        sys.exit(1)
    main(sys.argv[1])
