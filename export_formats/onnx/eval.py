
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



from matrice.actionTracker import ActionTracker
from matrice.matrice import Session


import onnx
import onnxruntime

def main(action_id):

    session=Session()
    actionTracker = ActionTracker(session,action_id)
    
    stepCode='MDL_EVL_ACK'
    status='OK'
    status_description='Model Evaluation has acknowledged'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)
    
    model_config=actionTracker.get_job_params()
    actionTracker.download_model("model.onnx",model_type="exported")
    
    print('model_config is' ,model_config)

    model_config.batch_size=1
    model_config.workers=4
    model_config.data=f"workspace/{model_config['dataset_path']}/images"

    model = onnxruntime.InferenceSession("model.onnx", None) #onnx.load("model.onnx")
    
    train_loader, val_loader, test_loader = load_data(model_config)
    
    
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


def get_metrics(split, data_loader, onnx_model, index_to_labels):

    def run_validate_onnx(split, loader, session):

        all_outputs = []
        all_targets = []
        all_predictions = []

        end = time.time()

        for i, (images, target) in enumerate(loader):
            

            input_data = {"images": images.numpy()}  # Assuming the input name is 'input'

            output = session.run(None, input_data)
            predictions = torch.argmax(torch.tensor(output[0]), dim=1)

            all_predictions.append(predictions)
            all_outputs.append(torch.tensor(output[0]))  # Convert to PyTorch tensor
            all_targets.append(target)

        all_predictions = torch.cat(all_predictions, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = get_evaluation_results(split, all_predictions, all_outputs, all_targets, index_to_labels)

        return metrics

        
    metrics = run_validate_onnx(split, data_loader,onnx_model)

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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 eval.py <action_status_id>")
        sys.exit(1)
    action_status_id = sys.argv[1]
    main(action_status_id)