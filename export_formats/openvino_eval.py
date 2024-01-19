import numpy as np
from pathlib import Path
#import cv2
import matplotlib.pyplot as plt
import openvino as ov
import ipywidgets as widgets
import urllib.request
import numpy as np
import urllib.request
import json
import time
import os
from PIL import Image
import tarfile
import torch
import os
import argparse


urllib.request.urlretrieve(
    url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
    filename='notebook_utils.py'
)
from data_processing.export_formats.notebook_utils import download_file

def download_model_and_data(model_file_name,model_name):
    base_artifacts_dir = Path('./artifacts').expanduser()
    model_xml_name = f'{model_file_name}.xml'
    model_bin_name = f'{model_file_name}.bin'
    model_xml_path = base_artifacts_dir / model_xml_name

    base_url = f'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/{model_name}/FP32/'

    if not model_xml_path.exists():
        download_file(base_url + model_xml_name, model_xml_name, base_artifacts_dir)
        download_file(base_url + model_bin_name, model_bin_name, base_artifacts_dir)
    else:
        print(f'{model_name} already downloaded to {base_artifacts_dir}')

    return model_xml_path 
def select_inference_device():
    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    return device

def load_model(model_xml_path, device):
    core = ov.Core()
    model = core.read_model(model=model_xml_path)
    compiled_model = core.compile_model(model=model, device_name=device.value)
    output_layer = compiled_model.output(0)
    return compiled_model, output_layer

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)



def do_inference(compiled_model, output_layer, input_image):
    output = compiled_model([input_image])[output_layer]
    result_index = np.argmax(output)
    return output,result_index

from python_common.model_utils.metrics.classificationmetrics import accuracy, precision, recall, f1_score,specificity


def get_image_files(directory):
    all_files = os.listdir(directory)
    image_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    return image_files

def evaluate_images(compiled_model,output_layer, labels, image_directory):
    class_folders = [f for f in os.listdir(image_directory) if os.path.isdir(os.path.join(image_directory, f))]

    actual_labels = []
    predicted_labels = []

    for class_folder in class_folders:
        class_path = os.path.join(image_directory, class_folder)
        image_files = get_image_files(class_path)

        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            image = Image.open(image_path).convert("RGB")
            input_image = image.resize((224, 224))
            input_image_array = np.array(input_image)

            input_image = np.expand_dims(input_image_array, 0)
            
            
            output,result_index = do_inference(compiled_model, output_layer, input_image)
            predicted_labels.append(output)
            predicted_label=labels[result_index]
            print(predicted_label)
            
            target_labels=[285,285,285,281,207,158,167,167,235] #Note: this is dummy model with a dummy dataset whose labels do not match-test split


   
    numeric_actual_labels = np.array(target_labels)
    numeric_predicted_labels = np.array(predicted_labels)
    print(numeric_predicted_labels)
    print(numeric_actual_labels)

    true_labels = torch.tensor(numeric_actual_labels)
    predicted_labels = torch.tensor(numeric_predicted_labels)
    print(predicted_labels.shape)
    print(true_labels.shape)
    metrics={}
    metrics['acc1'] = accuracy(predicted_labels, true_labels, topk=(1,))[0]
    metrics['acc5'] = accuracy(predicted_labels, true_labels, topk=(5,))[0]
    metrics['precision'] = precision(predicted_labels, true_labels)
    metrics['recall'] = recall(predicted_labels, true_labels)
    metrics['f1_score'] = f1_score(predicted_labels, true_labels)
    metrics['specificity'] = specificity(predicted_labels, true_labels)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")
    
    
def main(splits):


    labels = load_labels('imagenet-simple-labels.json')#from actiontracker



    for split in splits:
        split_path = os.path.join(r"workspace\6593b7ae10edfa25bc17cfe7-v1.0-imagenet\images", split)
        print(split_path)
        if not os.path.exists(split_path):
            print(f"Warning: Split {split} does not exist.")
            continue

        model_name = 'mobelinet-v3-tf'
        model_file_name='v3-small_224_1.0_float'
        model_xml_path = download_model_and_data(model_file_name,model_name)
            
        device = select_inference_device()
        compiled_model, output_layer = load_model(model_xml_path, device)
        evaluate_images(compiled_model,output_layer,labels,split_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONNX Evaluation Script')
    parser.add_argument('splits', metavar='SPLIT', nargs='+',
                            help='List of splits to evaluate')

    args = parser.parse_args()
    main(args.splits)