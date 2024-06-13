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
from python_common.services.utils import log_error



from matrice_sdk.actionTracker import ActionTracker
from train import load_data, update_compute



def main(action_id):

    
    try:
        actionTracker = ActionTracker(action_id)
    except Exception as e:
        log_error(__file__, 'main', f'Error initializing ActionTracker: {str(e)}')
        print(f"Error initializing ActionTracker: {str(e)}")
        sys.exit(1)
    
    try:
        actionTracker.update_status('MDL_EVL_ACK', 'OK', 'Model Evaluation has acknowledged')
    except Exception as e:
        actionTracker.update_status('MDL_EVL_ERR', 'ERROR', f'Error in starting evaluation: {str(e)}')
        log_error(__file__, 'main', f'Error updating status to MDL_EVL_ACK: {str(e)}')
        print(f"Error updating status to MDL_EVL_ACK: {str(e)}")
        sys.exit(1)
    
    try:
        actionTracker.download_model('model.pt')
        print('model_config is' ,actionTracker.model_config)
        actionTracker.model_config.data = f"workspace/{actionTracker.model_config['dataset_path']}/images"
        model = torch.load('model.pt', map_location='cpu')
        actionTracker.model_config.batch_size=32
        actionTracker.model_config.workers=4
        train_loader, val_loader, test_loader = load_data(actionTracker.model_config)
        device = update_compute(model)
        criterion = nn.CrossEntropyLoss().to(device)
        actionTracker.update_status('MDL_EVL_STRT','OK','Model Evaluation has started')
    except Exception as e:
        actionTracker.update_status('MDL_EVL_ERR', 'ERROR', f'Error in starting Evaluation: {str(e)}')
        log_error(__file__, 'main', f'Error updating status to MDL_EVL_STRT: {str(e)}')
        print(f"Error updating status to MDL_EVL_STRT: {str(e)}")
        sys.exit(1)

    try:
        index_to_labels=actionTracker.get_index_to_category()
        payload=[]
        
        if 'train' in actionTracker.model_config.split_types and os.path.exists(os.path.join(actionTracker.model_config.data, 'train')):
            payload+=get_metrics('train',train_loader, model,index_to_labels)

        if 'val' in actionTracker.model_config.split_types and os.path.exists(os.path.join(actionTracker.model_config.data, 'val')):
            payload+=get_metrics('val',val_loader, model,index_to_labels)


        if 'test' in actionTracker.model_config.split_types and os.path.exists(os.path.join(actionTracker.model_config.data, 'test')):
            payload+=get_metrics('test',test_loader, model,index_to_labels)


        actionTracker.save_evaluation_results(payload)
        actionTracker.update_status('MDL_EVL_CMPL','SUCCESS','Model Evaluation is completed')
        print(payload)
    except Exception as e:
        actionTracker.update_status('MDL_EVL_ERR', 'ERROR', f'Error in completing Evaluation: {str(e)}')
        log_error(__file__, 'main', f'Error updating status to MDL_EVL_CMPL: {str(e)}')
        print(f"Error updating status to MDL_EVL_CMPL: {str(e)}")
        sys.exit(1)    


from model_metrics import accuracy,precision,recall,f1_score_per_class,specificity,calculate_metrics_for_all_classes,specificity_all

def get_evaluation_results(split,predictions,output,target,index_to_labels):
    
        results=[]

        acc1 = accuracy(output, target, topk=(1,))[0]

        try:
            acc5 = accuracy(output, target, topk=(5,))[0]
        except:
            if torch.cuda.is_available():
                acc5 = torch.tensor([1])
            else:
                acc5 = torch.tensor([1])

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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 eval.py <action_status_id>")
        sys.exit(1)
    action_status_id = sys.argv[1]
    main(action_status_id)
