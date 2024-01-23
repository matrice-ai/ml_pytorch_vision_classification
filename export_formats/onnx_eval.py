import numpy as np
import onnxruntime
import onnx
from onnx import numpy_helper
import urllib.request
import json
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
import tarfile
import torch
import os
import argparse

import os
from PIL import Image

def download_file(url, filename):
    urllib.request.urlretrieve(url, filename)

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

def load_onnx_model(model_path):
    return onnxruntime.InferenceSession(model_path, None)


def preprocess(input_data):
    img_data = input_data.astype('float32')
    print(img_data.shape)
    img = Image.fromarray(np.uint8(img_data.transpose(1, 2, 0)))
    img = img.resize((224, 224))
    img_data_resized = np.array(img).transpose(2, 0, 1)
    print(img_data_resized.shape)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data_resized.shape).astype('float32')
    for i in range(img_data_resized.shape[0]):
        norm_img_data[i,:,:] = (img_data_resized[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()

def run_inference(session,input_name,input_data):
    start_time = time.time()
    raw_result = session.run([], {input_name:input_data})
    end_time = time.time()
    inference_time = np.round((end_time - start_time) * 1000, 2)
    return postprocess(raw_result), inference_time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from python_common.model_utils.metrics.classificationmetrics import accuracy, precision, recall, f1_score,specificity


def get_image_files(directory):
    all_files = os.listdir(directory)
    image_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    return image_files

def evaluate_images(session, input_name, labels, image_directory):
    class_folders = [f for f in os.listdir(image_directory) if os.path.isdir(os.path.join(image_directory, f))]

    actual_labels = []
    predicted_labels = []

    for class_folder in class_folders:
        class_path = os.path.join(image_directory, class_folder)
        image_files = get_image_files(class_path)

        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            image = Image.open(image_path)
            image_data = np.array(image).transpose(2, 0, 1)
            input_data = preprocess(image_data)

            outputs, inference_time = run_inference(session, input_name, input_data)

            predicted_idx = np.argmax(outputs)
            predicted_label = labels[predicted_idx]
            print(predicted_idx,predicted_label)
            predicted_labels.append(outputs)
            
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
    onnx_model_url = "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz"
    imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

    download_file(onnx_model_url, "resnet50v2.tar.gz")
    download_file(imagenet_labels_url, "imagenet-simple-labels.json")
    with tarfile.open("resnet50v2.tar.gz", "r:gz") as tar:
        tar.extractall()


    labels = load_labels('imagenet-simple-labels.json')#from actiontracker
    onnx_model_path = 'resnet50v2/resnet50v2.onnx' #from s3
    session = load_onnx_model(onnx_model_path)


    for split in splits:
        split_path = os.path.join(r"workspace\6593b7ae10edfa25bc17cfe7-v1.0-imagenet\images", split)
        print(split_path)
        if not os.path.exists(split_path):
            print(f"Warning: Split {split} does not exist.")
            continue

        onnx_model_path = 'resnet50v2/resnet50v2.onnx'
        session = onnxruntime.InferenceSession(onnx_model_path)
        input_name = session.get_inputs()[0].name
        evaluate_images(session,input_name,labels,split_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONNX Evaluation Script')
    parser.add_argument('splits', metavar='SPLIT', nargs='+',
                            help='List of splits to evaluate')

    args = parser.parse_args()
    main(args.splits)
