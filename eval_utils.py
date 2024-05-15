import numpy as np
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

import zipfile
from openvino.inference_engine import IECore
import time
import onnx
import onnxruntime

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

def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e., pinned memory)
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    return h_input, d_input, h_output, d_output, stream


def do_inference(engine, context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # Run inference
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

    # Transfer predictions back from device
    cuda.memcpy_dtoh_async(h_output, d_output, stream)

    # Synchronize the stream
    stream.synchronize()

    return h_output


def load_engine(trt_filename):
    # Deserialize the engine from file
    with open(trt_filename, 'rb') as f:
        engine_data = f.read()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_data)
    print("Engine:", engine)
    return engine


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



def get_model(framework):
    if framework=='ONNX':
        path_to_model="model.onnx"
    elif framework=='TensorRT':
        path_to_model="model.engine"
    elif framework=='OpenVINO':
        path_to_model="model_openvino.zip"
    elif framework=='TorchScript':
        path_to_model='model.torchscript'
    return path_to_model


def load_model(framework):
    if framework=='ONNX':
        model = onnxruntime.InferenceSession("model.onnx", None)
    elif framework=='TensorRT':
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        model = load_engine("model.engine")
    elif framework=='OpenVINO':
        with zipfile.ZipFile("model_openvino.zip", 'r') as zip_ref:
            zip_ref.extractall("/model_openvino")

        model_xml_path="/model_openvino/model.xml"
        # Initialize OpenVINO Inference Engine
        ie = IECore()
        device_name = 'GPU' if 'GPU' in ie.available_devices else 'CPU'
        # Load OpenVINO IR model on specified device (CPU or GPU)
        net = ie.read_network(model=model_xml_path)
        model = ie.load_network(network=net, device_name=device_name)
    elif framework=='TorchScript':
        model = torch.jit.load('model.torchscript', map_location='cpu')
    return model


def get_metrics(split,data_loader,framework,model,index_to_labels):
    if framework=='ONNX':

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

            
        metrics = run_validate_onnx(split, data_loader,model)

        

        
    elif framework=='TensorRT':
        def run_validate(split, loader, engine):
            context = engine.create_execution_context()
            h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)

            all_outputs = []
            all_targets = []
            all_predictions = []

            end = time.time()

            for i, (images, target) in enumerate(loader):
                images = images.numpy().astype(np.float32)  # Assuming your images are in the correct format
                h_input = np.ravel(images)

                # Run inference and get predictions
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


        metrics = run_validate(split, data_loader,model)

        
    elif framework=='OpenVINO':
        def run_validate(split, loader, net):
            all_outputs = []
            all_targets = []
            all_predictions = []

            end = time.time()

            for i, (images, target) in enumerate(loader):
                input_blob = next(iter(net.input_info))
                images = images.transpose(2, 3).transpose(1, 2) # Change data layout to NCHW
                images= np.transpose(images, (0, 2, 1, 3))
                images = np.ascontiguousarray(images)

                # Perform inference
                res = net.infer(inputs={input_blob: images})

                # Process output
                output = res[next(iter(res))]
                predictions = np.argmax(output, axis=1)

                all_predictions.append(torch.tensor(predictions))
                all_outputs.append(torch.tensor(output))  # Convert to PyTorch tensor
                all_targets.append(target)

            all_predictions = torch.cat(all_predictions, dim=0)
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            metrics = get_evaluation_results(split, all_predictions, all_outputs, all_targets, index_to_labels)

            return metrics


        metrics = run_validate(split, data_loader, model) 
    elif framework=='TorchScript':
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
        model.eval()
    
        if torch.cuda.is_available():
            model= model.cuda(0)  
            
        metrics = run_validate(split, data_loader)
    return metrics