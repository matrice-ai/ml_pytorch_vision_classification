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
import numpy as np
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
import pycuda.driver as cuda




from matrice_sdk.actionTracker import ActionTracker
from train import load_data, update_compute



import sys
import subprocess
#from matrice_sdk.deploy import MatriceDeploy
from PIL import Image, ImageDraw
from io import BytesIO

from export_formats.openvino.predict import load_model as load_openvino, predict as predict_openvino
from export_formats.torchscript.predict import load_model as load_torchscript, predict as predict_torchscript
from export_formats.onnx.predict import load_model as load_onnx
from predict import load_model as load_pytorch, predict as predict_pytorch

def load_model(actionTracker):
    runtime_framework=actionTracker.action_details['runtimeFramework'].lower()
    try:
        if "onnx" in runtime_framework:
            model=load_onnx(actionTracker)
        elif "torchscript" in runtime_framework:
            model=load_torchscript(actionTracker)
        elif "pytorch" in runtime_framework:
            model=load_pytorch(actionTracker)
        elif "tensorrt" in runtime_framework:
            subprocess.run(["pip", "install", "tensorrt", "pycuda"]) # Very large to add to requirements.txt
            from export_formats.tensorrt.predict import load_model as load_tensorrt, predict as predict_tensorrt
            model=load_tensorrt(actionTracker)
        elif "openvino" in runtime_framework:
            model=load_openvino(actionTracker)
        print(runtime_framework)
    except Exception as e:
        model=load_pytorch(actionTracker)
        print(f"ERROR: {e}")
        actionTracker.update_status('MDL_LOAD_ERR', 'ERROR', f'Error in model loading: {str(e)}')
        sys.exit(1)
    model_data={
        "framework": runtime_framework,
        "model": model
        }
    
    return model_data


def main(action_id):

    # Initializing the ActionTracker
    try:
        actionTracker = ActionTracker(action_id)
    except Exception as e:
        actionTracker.log_error(__file__, 'ml_pytorch_vision_classification/main', f'Error initializing ActionTracker: {str(e)}')
        print(f"Error initializing ActionTracker: {str(e)}")
        sys.exit(1)
    
    # Starting model evaluation
    try:
        actionTracker.update_status('MDL_EVL_ACK', 'OK', 'Model Evaluation has acknowledged')
    except Exception as e:
        actionTracker.update_status('MDL_EVL_ERR', 'ERROR', f'Error in starting evaluation: {str(e)}')
        actionTracker.log_error(__file__, 'ml_pytorch_vision_classification/main', f'Error updating status to MDL_EVL_ACK: {str(e)}')
        print(f"Error updating status to MDL_EVL_ACK: {str(e)}")
        sys.exit(1)
    
    # Loading Test Data   
    try:
        actionTracker.model_config.data = f"workspace/{actionTracker.model_config['dataset_path']}/images"
        val_loader, test_loader = load_data(actionTracker.model_config) 
        actionTracker.udpate_status('MDL_EVL_DTL', 'OK', 'Testing dataset is loaded')  
        
    except Exception as e:
        actionTracker.update_status('MDL_EVL_ERR', 'ERROR', f'Error in loading dataset: {str(e)}')
        actionTracker.log_error(__file__, 'ml_pytorch_vision_classification/main', f'Error updating status to MDL_EVL_DTL: {str(e)}')
        print(f"Error updating status to MDL_EVL_DTL: {str(e)}")
        sys.exit(1)
    
    # Loading model
    try:
        actionTracker.download_model('model.pt')
        print('model_config is' ,actionTracker.model_config)


        model_details=load_model(actionTracker)
        model=model_details['model']
        #model = torch.load('model.pt', map_location='cpu')
        framework=model_details['framework']
        actionTracker.model_config.batch_size=32
        actionTracker.model_config.workers=4
        device = update_compute(model)
        criterion = nn.CrossEntropyLoss().to(device)
        actionTracker.update_status('MDL_EVL_STRT','OK','Model Evaluation has started')
        
    except Exception as e:
        actionTracker.update_status('MDL_EVL_ERR', 'ERROR', f'Error in starting Evaluation: {str(e)}')
        actionTracker.log_error(__file__, 'ml_pytorch_vision_classification/main', f'Error updating status to MDL_EVL_STRT: {str(e)}')
        print(f"Error updating status to MDL_EVL_STRT: {str(e)}")
        sys.exit(1)

    # Evaluating on test dataset
    try:
        index_to_labels=actionTracker.get_index_to_category()
        payload=[]
        
        if 'val' in actionTracker.model_config.split_types and os.path.exists(os.path.join(actionTracker.model_config.data, 'val')):
            payload+=get_metrics('val',val_loader, framework, model,index_to_labels)

        if 'test' in actionTracker.model_config.split_types and os.path.exists(os.path.join(actionTracker.model_config.data, 'test')):
            payload+=get_metrics('test',test_loader, framework, model,index_to_labels)

        actionTracker.save_evaluation_results(payload)
        actionTracker.update_status('MDL_EVL_CMPL','SUCCESS','Model Evaluation is completed')
        print(payload)
        
    except Exception as e:
        actionTracker.update_status('MDL_EVL_ERR', 'ERROR', f'Error in completing Evaluation: {str(e)}')
        actionTracker.log_error(__file__, 'ml_pytorch_vision_classification/main', f'Error updating status to MDL_EVL_CMPL: {str(e)}')
        print(f"Error updating status to MDL_EVL_CMPL: {str(e)}")
        sys.exit(1)    


from model_metrics import accuracy, precision, recall, f1_score_per_class, specificity, calculate_metrics_for_all_classes, specificity_all, accuracy_per_class

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

        for name,value in accuracy_per_class(output,target).items():
            results.append({
            "category": index_to_labels[str(name)],
             "splitType":split,
             "metricName":"acc@1",
            "metricValue":float(value)
         })
            
        return results


def allocate_buffers(engine):
    import tensorrt as trt
    # Get input and output binding indices (assuming 0 and 1, but verify if there are multiple inputs/outputs)
    input_binding_index = 0
    output_binding_index = 1

    # Determine dimensions and create page-locked memory buffers (i.e., pinned memory)
    input_shape = engine.get_binding_shape(input_binding_index)
    output_shape = engine.get_binding_shape(output_binding_index)

    # Volume calculation assumes fully defined shapes. Adjust if dimensions are dynamic.
    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    return h_input, d_input, h_output, d_output, stream


def do_inference(engine, context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # Run inference asynchronously
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

    # Transfer predictions back from device
    cuda.memcpy_dtoh_async(h_output, d_output, stream)

    # Synchronize the stream to ensure the inference is completed before accessing output
    stream.synchronize()

    return h_output


def get_metrics(split, data_loader, framework, model, index_to_labels):
    
    def run_validate_pytorch(split, loader):
        all_outputs = []
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                if torch.cuda.is_available():
                    images = images.cuda(0, non_blocking=True)
                    target = target.cuda(0, non_blocking=True)

                output = model(images)
                predictions = torch.argmax(output, dim=1)

                all_predictions.append(predictions)
                all_outputs.append(output)
                all_targets.append(target)

            all_predictions = torch.cat(all_predictions, dim=0)
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            metrics = get_evaluation_results(split, all_predictions, all_outputs, all_targets, index_to_labels)

            return metrics
    
    framework = framework.lower()  # Convert framework to lowercase
    
    if framework == 'pytorch':
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda(0)
        metrics = run_validate_pytorch(split, data_loader)

    elif framework == 'torchscript':
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda(0)
        metrics = run_validate_pytorch(split, data_loader)
    
    elif framework == 'onnx':
        def run_validate_onnx(split, loader, session):
            all_outputs = []
            all_targets = []
            all_predictions = []

            for i, (images, target) in enumerate(loader):
                input_data = {"images": images.numpy()}  # Adjust input name if needed
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
            
        metrics = run_validate_onnx(split, data_loader, model)
        
    elif framework == 'tensorrt':
        def run_validate_tensorrt(split, loader, engine):
            context = engine.create_execution_context()
            h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)

            all_outputs = []
            all_targets = []
            all_predictions = []

            for i, (images, target) in enumerate(loader):
                images = images.numpy().astype(np.float32)
                h_input = np.ravel(images)

                output = do_inference(engine, context, h_input, d_input, h_output, d_output, stream)
                predictions = np.argmax(output)

                all_predictions.append(predictions)
                all_outputs.append(output)
                all_targets.append(target)

            all_predictions = torch.tensor(all_predictions)
            all_outputs = torch.tensor(all_outputs)
            all_targets = torch.tensor(all_targets)

            metrics = get_evaluation_results(split, all_predictions, all_outputs, all_targets, index_to_labels)

            return metrics
        
        metrics = run_validate_tensorrt(split, data_loader, model)

    elif framework == 'openvino':
        def run_validate_openvino(split, loader, net):
            all_outputs = []
            all_targets = []
            all_predictions = []

            for i, (images, target) in enumerate(loader):
                input_blob = next(iter(net.input_info))
                images = images.transpose(2, 3).transpose(1, 2)  # NCHW layout
                images = np.ascontiguousarray(images)

                res = net.infer(inputs={input_blob: images})
                output = res[next(iter(res))]
                predictions = np.argmax(output, axis=1)

                all_predictions.append(torch.tensor(predictions))
                all_outputs.append(torch.tensor(output))
                all_targets.append(target)

            all_predictions = torch.cat(all_predictions, dim=0)
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            metrics = get_evaluation_results(split, all_predictions, all_outputs, all_targets, index_to_labels)

            return metrics
        
        metrics = run_validate_openvino(split, data_loader, model)

    return metrics


# def get_metrics(split, data_loader, model, index_to_labels):

#     def run_validate(split, loader):
#         all_outputs = []
#         all_targets = []
#         all_predictions=[]
#         with torch.no_grad():
#             end = time.time()

#             for i, (images, target) in enumerate(loader):
#                 if torch.cuda.is_available():
#                     images = images.cuda(0, non_blocking=True)
#                     target = target.cuda(0, non_blocking=True)

#                 output = model(images)
#                 predictions = torch.argmax(output, dim=1)

#                 all_predictions.append(predictions)
#                 all_outputs.append(output)
#                 all_targets.append(target)

#             all_predictions= torch.cat(all_predictions, dim=0)
#             all_outputs = torch.cat(all_outputs, dim=0)
#             all_targets = torch.cat(all_targets, dim=0)

#             metrics = get_evaluation_results(split,all_predictions, all_outputs, all_targets, index_to_labels)

#             return metrics

    # switch to evaluate mode
    # model.eval()
    
    # if torch.cuda.is_available():
    #     model= model.cuda(0)  
        
    # metrics = run_validate(split, data_loader)

    # return metrics


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 eval.py <action_status_id>")
        sys.exit(1)
    action_status_id = sys.argv[1]
    main(action_status_id)
