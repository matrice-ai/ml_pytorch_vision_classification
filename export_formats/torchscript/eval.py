import argparse
import os
import random
import shutil
import time
import warnings
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


def main(action_id):

    session=Session()
    actionTracker = ActionTracker(session,action_id)
    
    stepCode='MDL_EVL_ACK'
    status='OK'
    status_description='Model Evaluation has acknowledged'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)
    
    model_config=actionTracker.get_job_params()
    actionTracker.download_model('model.torchscript',model_type="exported")
    
    print('model_config is' ,model_config)

    model_config.batch_size=32
    model_config.workers=4
    _idDataset=model_config['_idDataset']
    dataset_version=model_config['dataset_version']
    model_config.data=f'workspace/{str(_idDataset)}-{str(dataset_version).lower()}-imagenet/images'

    model = torch.jit.load('model.torchscript', map_location='cpu')
    
    train_loader, val_loader, test_loader = load_data(model_config)
    
    device = update_compute(model)
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    stepCode='MDL_EVL_STRT'
    status='OK'
    status_description='Model Evaluation has started'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)

    index_to_labels=actionTracker.get_index_to_category(is_exported=True)
    payload=[]
    
    if 'train' in model_config.split_types and os.path.exists(os.path.join(model_config.data, 'train')):
        payload+=get_metrics('train',train_loader, model,index_to_labels)

    if 'val' in model_config.split_types and os.path.exists(os.path.join(model_config.data, 'val')):
        payload+=get_metrics('val',val_loader, model,index_to_labels)


    if 'test' in model_config.split_types and os.path.exists(os.path.join(model_config.data, 'test')):
        payload+=get_metrics('test',test_loader, model,index_to_labels)


    actionTracker.save_evaluation_results(payload)
    
    stepCode='MDL_EVL_CMPL'
    status='SUCCESS'
    status_description='Model Evaluation is completed'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)
    
    print(payload)


def get_metrics(split, data_loader, model, index_to_labels):

    def run_validate(split, loader):
        all_outputs = []
        all_targets = []
        all_predictions=[]
        with torch.no_grad():
            end = time.time()

            for i, (images, target) in enumerate(loader):
                if torch.cuda.is_available():
                    images = images.cuda(0, non_blocking=True)
                    target = target.cuda(0, non_blocking=True)

                output = model(images)
                predictions = torch.argmax(output, dim=1)

                all_predictions.append(predictions)
                all_outputs.append(output)
                all_targets.append(target)

            all_predictions= torch.cat(all_predictions, dim=0)
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            metrics = get_evaluation_results(split,all_predictions, all_outputs, all_targets, index_to_labels)

            return metrics

    # switch to evaluate mode
    model.eval()
    
    if torch.cuda.is_available():
        model= model.cuda(0)  
        
    metrics = run_validate(split, data_loader)

    return metrics

from model_metrics import accuracy,precision,recall,f1_score_per_class,specificity,calculate_metrics_for_all_classes,specificity_all

def get_evaluation_results(split,predictions,output,target,index_to_labels):
    
        results=[]

        acc1 = accuracy(output, target, topk=(1,))[0]

        try:
            acc5 = accuracy(output, target, topk=(5,))[0]
        except:
            if torch.cuda.is_available():
                acc5 = torch.tensor([100])
            else:
                acc5 = torch.tensor([100])

        precision_all, recall_all, f1_score_all=calculate_metrics_for_all_classes(predictions,target)
        
        results.append({
            "category":"all",
             "splitType":split,
             "metricName":"acc@1",
            "metricValue":float(acc1.item())
         })

        results.append({
            "category":"all",
             "splitType":split,
             "metricName":"acc@5",
            "metricValue":float(acc5.item())
         })

        results.append({
            "category":"all",
             "splitType":split,
             "metricName":"specificity",
            "metricValue":float(specificity_all(output,target))
         })
        
        results.append({
            "category":"all",
             "splitType":split,
             "metricName":"precision",
            "metricValue":float(precision_all)
         })
        
        results.append({
            "category":"all",
             "splitType":split,
             "metricName":"recall",
            "metricValue":float(recall_all)
         })
        
        results.append({
            "category":"all",
             "splitType":split,
             "metricName":"f1_score",
            "metricValue":float(f1_score_all)
         })
                
        for name,value in precision(output,target).items():
            results.append({
            "category": index_to_labels[str(name)],
             "splitType":split,
             "metricName":"precision",
            "metricValue":float(value)
         })


        for name,value in f1_score_per_class(output,target).items():
            results.append({
            "category": index_to_labels[str(name)],
             "splitType":split,
             "metricName":"f1_score",
            "metricValue":float(value)
         })

        for name,value in recall(output,target).items():
            results.append({
            "category": index_to_labels[str(name)],
             "splitType":split,
             "metricName":"recall",
            "metricValue":float(value)
         })

        for name,value in specificity(output,target).items():
            results.append({
            "category": index_to_labels[str(name)],
             "splitType":split,
             "metricName":"specificity",
            "metricValue":float(value)
         })

        return results
        
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 eval.py <action_status_id>")
        sys.exit(1)
    action_status_id = sys.argv[1]
    main(action_status_id)
