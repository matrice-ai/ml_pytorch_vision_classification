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

from matrice_sdk.actionTracker import ActionTracker
from matrice_sdk.matrice import Session
from train import load_data, update_compute
from model_metrics import accuracy, precision, recall, f1_score_per_class, specificity, calculate_metrics_for_all_classes, specificity_all


def main(action_id):
    session = Session()
    actionTracker = ActionTracker(session, action_id)

    stepCode, status = 'MDL_EVL_ACK', 'OK'
    status_description = 'Model Evaluation has acknowledged'
    print(status_description)
    actionTracker.update_status(stepCode, status, status_description)

    model_config = actionTracker.get_job_params()
    actionTracker.download_model('model.pt')
    print('model_config is', model_config)

    model_config.data = f"workspace/{model_config['dataset_path']}/images"
    model = torch.load('model.pt', map_location='cpu')
    model_config.batch_size, model_config.workers = 32, 4
    train_loader, val_loader, test_loader = load_data(model_config)
    device = update_compute(model)

    criterion = nn.CrossEntropyLoss().to(device)

    stepCode, status_description = 'MDL_EVL_STRT', 'Model Evaluation has started'
    print(status_description)
    actionTracker.update_status(stepCode, status, status_description)

    index_to_labels = actionTracker.get_index_to_category()
    payload = []

    # Combine the if conditions and add the results to the payload
    for split_type, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        if split_type in model_config.split_types and os.path.exists(os.path.join(model_config.data, split_type)):
            payload += get_metrics(split_type, loader, model, index_to_labels)

    actionTracker.save_evaluation_results(payload)

    stepCode, status, status_description = 'MDL_EVL_CMPL', 'SUCCESS', 'Model Evaluation is completed'
    print(status_description)
    actionTracker.update_status(stepCode, status, status_description)

    print(payload)


def get_evaluation_results(split, predictions, output, target, index_to_labels):
    results = []
    acc1 = accuracy(output, target, topk=(1,))[0]
    acc5 = accuracy(output, target, topk=(5,))[0] if output.size(1) >= 5 else torch.tensor([1])

    precision_all, recall_all, f1_score_all = calculate_metrics_for_all_classes(predictions, target)

    metrics_all = {
        "acc@1": float(acc1.item()),
        "acc@5": float(acc5.item()),
        "specificity": float(specificity_all(output, target)),
        "precision": float(precision_all),
        "recall": float(recall_all),
        "f1_score": float(f1_score_all)
    }

    for metric_name, metric_value in metrics_all.items():
        results.append({
            "category": "all",
            "splitType": split,
            "metricName": metric_name,
            "metricValue": metric_value
        })

    # Simplify adding per-class metrics by combining in a loop
    for metric_fn, metric_name in [(precision, "precision"), (recall, "recall"), (f1_score_per_class, "f1_score"), (specificity, "specificity")]:
        for name, value in metric_fn(output, target).items():
            results.append({
                "category": index_to_labels[str(name)],
                "splitType": split,
                "metricName": metric_name,
                "metricValue": float(value)
            })

    return results


def get_metrics(split, data_loader, model, index_to_labels):
    # switch to evaluate mode
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda(0)

    def run_validate(loader):
        all_outputs, all_targets, all_predictions = [], [], []
        with torch.no_grad():
            for images, target in loader:
                if torch.cuda.is_available():
                    images, target = images.cuda(0, non_blocking=True), target.cuda(0, non_blocking=True)

                output = model(images)
                predictions = torch.argmax(output, dim=1)

                all_predictions.append(predictions)
                all_outputs.append(output)
                all_targets.append(target)

        all_predictions = torch.cat(all_predictions, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        return get_evaluation_results(split, all_predictions, all_outputs, all_targets, index_to_labels)

    return run_validate(data_loader)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 eval.py <action_status_id>")
        sys.exit(1)
    main(sys.argv[1])   
