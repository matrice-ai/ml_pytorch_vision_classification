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
from  matrice_sdk import rpc
from matrice_sdk.actionTracker import ActionTracker, LocalActionTracker
import traceback



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


best_acc1 = 0


def main(action_id=None):
    global best_acc1
    global actionTracker
    
    try:
        if action_id:
            actionTracker = ActionTracker(action_id)
        else:
            action_type = input("Enter the action type : Train/Export : ")
            model_name = input("Enter the name of model family : ")
            model_arch = input("Enter the model Architecture , enter existing architecture only! : ")
            output_type = input("Enter the type of output - detection / classification : ")
            actionTracker = LocalActionTracker(action_type , model_name , model_arch , output_type)
    except Exception as e:
        actionTracker.log_error(__file__, 'ml_pytorch_vision_classification/main', f'Error initializing ActionTracker: {str(e)}')
        print(f"Error initializing ActionTracker: {str(e)}")
        sys.exit(1)
    
    # Acknowledging the model training
    try:
        actionTracker.update_status('MDL_TRN_ACK', 'OK', 'Model Training has acknowledged')
    except Exception as e:
        actionTracker.update_status('MDL_TRN_ERR', 'ERROR', f'Error in starting training: {str(e)}')
        actionTracker.log_error(__file__, 'ml_pytorch_vision_classification/main', f'Error updating status to MDL_TRN_ACK: {str(e)}')
        print(f"Error updating status to MDL_TRN_ACK: {str(e)}")
        sys.exit(1)
    

    model_config = actionTracker.get_job_params()
    print('model_config is', model_config)

    # Loading the data
    try:
        model_config.data = f"workspace/{model_config['dataset_path']}/images"
        train_loader, val_loader, test_loader = load_data(model_config)
        index_to_labels = {str(idx): str(label) for idx, label in enumerate(train_loader.dataset.classes)}
        actionTracker.add_index_to_category(index_to_labels)
        actionTracker.update_status('MDL_TRN_DTL', 'OK', 'Training dataset is loaded')
        
    except Exception as e:
        actionTracker.update_status('MDL_TRN_DTL', 'ERROR', 'Error in loading training dataset')
        actionTracker.log_error(__file__, 'ml_pytorch_vision_classification/main', f'Error updating status to MDL_TRN_DTL: {str(e)}')
        print(f"Error updating status to MDL_TRN_DTL: {str(e)}")
        sys.exit(1)
    
    # Initializing model    
    try: 
        model = initialize_model(model_config, train_loader.dataset)
        device = update_compute(model)
        actionTracker.update_status('MDL_TRN_MDL', 'OK', 'Model has been loaded') 
    except Exception as e:
        actionTracker.update_status('MDL_TRN_MDL', 'ERROR', 'Error in loading model')
        actionTracker.log_error(__file__, 'ml_pytorch_vision_classification/main', f'Error updating status to MDL_TRN_MDL: {str(e)}')
        print(f"Error updating status to MDL_TRN_MDL: {str(e)}")
        sys.exit(1)
            
    # Setting up the training of the model
    try:
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = setup_optimizer(model, model_config)
        scheduler = setup_scheduler(optimizer, model_config)
        actionTracker.update_status('MDL_TRN_STRT', 'OK', 'Model training is starting')
        
    except Exception as e:
        actionTracker.update_status('MDL_TRN_SETUP', 'ERROR', 'Error in setting up model training')
        actionTracker.log_error(__file__, 'ml_pytorch_vision_classification/main', f'Error updating status to MDL_TRN_STRT: {str(e)}')
        print(f"Error updating status to MDL_TRN_STRT: {str(e)}")
        sys.exit(1)
        
    
    best_acc1 = -1.0
    best_model = None
    early_stopping=EarlyStopping(patience=model_config.patience,min_delta=model_config.min_delta)
    
    print("Starting Training")
    
    print(early_stopping)

    for epoch in range(model_config.epochs):
        
        print("Entered Training loop : " , {epoch})

        # train for one epoch
        loss_train,acc1_train, acc5_train =train(train_loader, model, criterion, optimizer, epoch, device, model_config)

        # evaluate on validation set
        loss_val,acc1_val,acc5_val= validate(val_loader, model, criterion,device, model_config)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1_val > best_acc1
        best_acc1 = max(acc1_val, best_acc1)
        
        # Logging training details   
        try:            
            epochDetails= [{"splitType": "train", "metricName": "loss", "metricValue":loss_train},
                            {"splitType": "train", "metricName": "acc@1", "metricValue": acc1_train},
                            {"splitType": "train", "metricName": "acc@5", "metricValue": acc5_train},
                            {"splitType": "val", "metricName": "loss", "metricValue": loss_val},
                            {"splitType": "val", "metricName": "acc@1", "metricValue": acc1_val},
                            {"splitType": "val", "metricName": "acc@5", "metricValue": acc5_val}]

            actionTracker.log_epoch_results(epoch ,epochDetails)
            print(epochDetails)
            
        except Exception as e:
            actionTracker.update_status('MDL_TRN_EPOCH', 'ERROR', 'Error in logging training epoch details')
            actionTracker.log_error(__file__, 'ml_pytorch_vision_classification/main', f'Error updating status to MDL_TRN_EPOCH: {str(e)}')
            print(f"Error updating status to MDL_TRN_EPOCH: {str(e)}")
            sys.exit(1) 
        
        if is_best:
            best_model = model
            save_checkpoint({
                'epoch': epoch,
                'arch': model_config.model_key,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, model,is_best)  
            


        early_stopping.update(loss_val)
        if early_stopping.stop:
            break

    

    # Saving the best model and checkpoint 
    try:
        ## For using as a checkpoint for training other models
        actionTracker.upload_checkpoint('checkpoints/model_best.pth.tar')
        
        ## For exporting, evaluation and deployment
        torch.save(best_model, 'checkpoints/model_best.pt')
        actionTracker.upload_checkpoint('checkpoints/model_best.pt')
    except:
        actionTracker.update_status('MDL_TRN_SBM', 'ERROR', 'Error in saving the best model')    
   
    # Evaluation of model
    from eval import get_metrics

    try:
        payload=[]
        ## Run on validation set
        if  os.path.exists(valdir):
            payload+=get_metrics('val',val_loader, best_model, index_to_labels)
            print(payload)
            
        ## Run on test set
        if  os.path.exists(testdir):
            payload+=get_metrics('test',test_loader, best_model, index_to_labels)
            print(payload)
        
        actionTracker.save_evaluation_results(payload)
        actionTracker.update_status('MDL_TRN_SUCCESS', 'SUCCESS', 'Model training is successfully completed')
    
    except Exception as e:
        actionTracker.update_status('MDL_TRN_EVAL', 'ERROR', 'Error in evaluation using the best model')
        actionTracker.log_error(__file__, 'ml_pytorch_vision_classification/main', f'Error updating status to MDL_TRN_EVAL: {str(e)}')
        print(f"Error updating status to MDL_TRN_EVAL: {str(e)}")
        sys.exit(1) 
            
    
    
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
    
    print("entered")
    global traindir, valdir, testdir
    traindir = os.path.join(model_config.data, 'train')
    valdir = os.path.join(model_config.data, 'val')
    testdir = os.path.join(model_config.data, 'test')
    
    print("entered")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    try:
        imsize = 299 if model_config.model_key.startswith('inception') else 224
    except Exception as e:
        # You can log the exception or print it if needed
        print(f"An error occurred: {e}")
        imsize = 224

    train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(imsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(imsize),
            transforms.CenterCrop(imsize),
            transforms.ToTensor(),
            normalize,
        ]))
   
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=model_config.batch_size, shuffle=False,
        num_workers=4)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=model_config.batch_size, shuffle=False,
        num_workers=4)
    
    
    
    test_loader = None
    if os.path.exists(testdir):
        test_dataset = datasets.ImageFolder(
            testdir,
            transforms.Compose([
                transforms.Resize(imsize),
                transforms.CenterCrop(imsize),
                transforms.ToTensor(),
                normalize,
            ]))
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=model_config.batch_size, shuffle=False,
            num_workers=4)

    return train_loader, val_loader, test_loader


def initialize_model(model_config, dataset):
    print("=> using pre-trained model '{}'".format(model_config.model_key))
    
    # Get the model function or class
    model_func = models.__dict__[model_config.model_key]
    
    # Check if it's a callable (function or class)
    if callable(model_func):
        if model_config.model_key == 'googlenet':
            model = model_func(pretrained=model_config.pretrained, aux_logits=False)
        elif model_config.model_key.startswith('inception'):
            model = model_func(pretrained=model_config.pretrained, aux_logits=False)    
        else:
            model = model_func(pretrained=model_config.pretrained)
    else:
        # If it's a module, we need to get the generating function
        model = getattr(model_func, model_config.model_key)(pretrained=model_config.pretrained)
    
    try:
        # Load checkpoint if available
        checkpoint_path, checkpoint_found = actionTracker.get_checkpoint_path(model_config)
        if checkpoint_found:
            print("Loading checkpoint from:", checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            print("Model loaded from checkpoint:", checkpoint_path)
        else:
            print("No checkpoint found. Using pre-trained or newly initialized weights.")

        # Modify the final layer
        if model_config.model_key.startswith('squeezenet'):
            # SqueezeNet-specific modification
            model.classifier[1] = nn.Conv2d(512, len(dataset.classes), kernel_size=(1,1), stride=(1,1))
            model.num_classes = len(dataset.classes)
        elif hasattr(model, 'fc'):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(dataset.classes))
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Linear):
                # For models like DenseNet where classifier is a single Linear layer
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, len(dataset.classes))
            elif isinstance(model.classifier, nn.Sequential):
                # For models where classifier is a Sequential module
                num_ftrs = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_ftrs, len(dataset.classes))
            else:
                raise AttributeError("Unexpected classifier structure")
        else:
            # For models with non-standard final layer structures
            if hasattr(model, 'avgpool') and hasattr(model, 'last_linear'):
                # This structure is common in some ResNet variants
                num_ftrs = model.last_linear.in_features
                model.last_linear = nn.Linear(num_ftrs, len(dataset.classes))
            else:
                raise AttributeError("Model structure not recognized")

        actionTracker.update_status('MDL_TRN_MDL', 'OK', 'Initial Model has been loaded')
        
        print(model)
        
    except Exception as e:
        print(f"Error in loading model: {str(e)}")
        actionTracker.update_status('MDL_TRN_MDL', 'ERROR', f'Error in loading model: {str(e)}')
    
    return model

def setup_optimizer(model, model_config):

    opt_name = model_config.optimizer.lower()

    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=model_config.learning_rate,
            momentum=model_config.momentum,
            weight_decay=model_config.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=model_config.learning_rate, momentum=model_config.momentum, weight_decay=model_config.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        print("Entering block 3")
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate, weight_decay=model_config.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {model_config.optimizer}. Only SGD, RMSprop and AdamW are supported.")
    
    
    print(optimizer)
    
    return optimizer

def setup_scheduler(optimizer, model_config):
    model_config.lr_scheduler = model_config.lr_scheduler.lower()
    print("Reached Scheduler")
    if model_config.lr_scheduler == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=model_config.lr_step_size, gamma=model_config.lr_gamma
        )
    elif model_config.lr_scheduler == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=model_config.epochs, eta_min=model_config.lr_min
        )
    elif model_config.lr_scheduler == "exponentiallr":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=model_config.lr_gamma
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{model_config.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR are supported."
        )
        
    print("Exited SCheduler")
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


import shutil
import copy

def save_checkpoint(state, model, is_best, filename='checkpoint.pth.tar'):
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Full path for the checkpoint file
    filepath = os.path.join(checkpoint_dir, filename)
    
    # Save the checkpoint
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

    if is_best:
        # Copy the best model
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        shutil.copyfile(filepath, best_filepath)
        print(f"Best model saved to {best_filepath}")

        # Save the best model in .pt format
        if isinstance(model, torch.nn.DataParallel):
            model_best = copy.deepcopy(model.module)
        else:
            model_best = model
        best_pt_filepath = os.path.join(checkpoint_dir, 'model_best.pt')
        torch.save(model_best, best_pt_filepath)
        print(f"Best model (PT format) saved to {best_pt_filepath}")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=10):
        self.patience = patience
        self.counter = 0
        self.lowest_loss = None
        self.min_delta = min_delta
        self.stop = False

    def update(self, val_loss):
        if self.lowest_loss is None:
            self.lowest_loss = val_loss

        elif val_loss < self.lowest_loss - self.min_delta:
            self.lowest_loss = val_loss
            self.counter = 0

        else:
            self.counter += 1
            print(f'Early stopping count is {self.counter}')
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('action_id', nargs='?', default=None, metavar='ACTION_ID', help='Action ID')
    args = parser.parse_args()
    main(args.action_id)
