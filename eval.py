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

#add Try Catch blocks

from python_sdk.src.actionTracker import ActionTracker
#from python_sdk.matrice import Session
from train import load_data, update_compute



def main(action_id):
    actionTracker = ActionTracker(action_id)
    try:
        stepCode='MDL_EVL_ACK'
        status='OK'
        status_description='Model Evaluation has acknowledged'
        print(status_description)
        actionTracker.update_status(stepCode,status,status_description) 
        #Above can be done in a single line
    except Exception as e:
        #Update status as error
        print(f"An error occurred: {e}")

    actionTracker.download_model('model.pt') #Inside a try catch block
    
    print('model_config is' ,actionTracker.model_config)

    _idDataset=actionTracker.model_config['_idDataset'] 
    dataset_version=actionTracker.model_config['dataset_version']
    actionTracker.model_config.data=f'workspace/{str(_idDataset)}-{str(dataset_version).lower()}-imagenet/images' #Get the images directly 

    model = torch.load('model.pt', map_location='cpu') #Try catch block 
    
    train_loader, val_loader, test_loader = load_data(model_config) #Try catch block
    
    device = update_compute(model) #Try catch block
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    stepCode='MDL_EVL_STRT'
    status='OK'
    status_description='Model Evaluation has started'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description) # COmplete the above in a single line

    index_to_labels=actionTracker.get_index_to_category()
    payload=[]
    
    if 'train' in actionTracker.model_config.split_types and os.path.exists(os.path.join(model_config.data, 'train')):
        payload+=get_metrics('train',train_loader, model,index_to_labels)

    if 'val' in actionTracker.model_config.split_types and os.path.exists(os.path.join(model_config.data, 'val')):
        payload+=get_metrics('val',val_loader, model,index_to_labels)


    if 'test' in actionTracker.model_config.split_types and os.path.exists(os.path.join(model_config.data, 'test')):
        payload+=get_metrics('test',test_loader, model,index_to_labels)
#Try catch block

    actionTracker.save_evaluation_results(payload) #Try catch block
    
    stepCode='MDL_EVL_CMPL'
    status='SUCCESS'
    status_description='Model Evaluation is completed'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description) # DOne in single line + Try catch block
    
    print(payload)


from model_metrics import accuracy,precision,recall,f1_score_per_class,specificity,calculate_metrics_for_all_classes,specificity_all #Move this to the start

def get_evaluation_results(split,predictions,output,target,index_to_labels):
    
        results=[]

        acc1 = accuracy(output, target, topk=(1,))[0]

        try:
            acc5 = accuracy(output, target, topk=(5,))[0]
        except: #Error logging
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

                output = model(images) #Try catch block
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 eval.py <action_status_id>")
        sys.exit(1)
    action_status_id = sys.argv[1]
    main(action_status_id)
