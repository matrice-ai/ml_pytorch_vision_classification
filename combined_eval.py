
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


from python_sdk.src.matrice_sdk.actionTracker import ActionTracker
from python_sdk.src.matrice_sdk.matrice import Session
from eval_utils import get_model,load_model,get_metrics



def main(action_id):

    session=Session()
    actionTracker = ActionTracker(session,action_id)
    framework=actionTracker.action_details["runtimeFramework"]
    
    stepCode='MDL_EVL_ACK'
    status='OK'
    status_description='Model Evaluation has acknowledged'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)
    
    model_config=actionTracker.get_job_params()
    path_to_model=get_model(framework)
    
    actionTracker.download_model(path_to_model,model_type="exported")
    
    print('model_config is' ,model_config)

    model_config.batch_size=1
    model_config.workers=4
    _idDataset=model_config['_idDataset']
    dataset_version=model_config['dataset_version']
    model_config.data=f'workspace/{str(_idDataset)}-{str(dataset_version).lower()}-imagenet/images'

    #model = onnxruntime.InferenceSession("model.onnx", None) #onnx.load("model.onnx")
    model=load_model(framework)
    train_loader, val_loader, test_loader = load_data(model_config)
    
    
    stepCode='MDL_EVL_STRT'
    status='OK'
    status_description='Model Evaluation has started'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)

    index_to_labels=actionTracker.get_index_to_category(is_exported=True)
    payload=[]
    
    if 'train' in model_config.split_types and os.path.exists(os.path.join(model_config.data, 'train')):
        payload+=get_metrics('train',train_loader, framework,model,index_to_labels)

    if 'val' in model_config.split_types and os.path.exists(os.path.join(model_config.data, 'val')):
        payload+=get_metrics('val',val_loader,framework,model,index_to_labels)


    if 'test' in model_config.split_types and os.path.exists(os.path.join(model_config.data, 'test')):
        payload+=get_metrics('test',test_loader,framework,model,index_to_labels)


    actionTracker.save_evaluation_results(payload)
    
    stepCode='MDL_EVL_CMPL'
    status='SUCCESS'
    status_description='Model Evaluation is completed'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)
    
    print(payload)




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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 eval.py <action_status_id>")
        sys.exit(1)
    action_status_id = sys.argv[1]
    main(action_status_id)