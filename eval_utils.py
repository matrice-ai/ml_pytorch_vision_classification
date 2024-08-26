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

import zipfile
from openvino.inference_engine import IECore
import time
import onnx
import onnxruntime


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

def get_onnx_inference_results(loader, model):

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

    return all_predictions, all_outputs, all_targets
            
def get_tensorrt_inference_results(loader, model):
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

    return all_predictions, all_outputs, all_targets

        

def get_openvino_inference_results(loader, model):
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
        all_outputs.append(torch.tensor(output))
        all_targets.append(target)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return all_predictions, all_outputs, all_targets
            
def get_torchscript_inference_results(loader, model):
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

        return all_predictions, all_outputs, all_targets
