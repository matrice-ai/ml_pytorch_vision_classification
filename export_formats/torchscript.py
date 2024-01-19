import numpy as np
import torch.nn.functional as F
from torchvision import transforms,models
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


def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

import numpy as np
from python_common.model_utils.metrics.classificationmetrics import accuracy, precision, recall, f1_score,specificity


def get_image_files(directory):
    all_files = os.listdir(directory)
    image_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    return image_files

def evaluate_images(model_path, labels, image_directory):
    class_folders = [f for f in os.listdir(image_directory) if os.path.isdir(os.path.join(image_directory, f))]

    predicted_labels = []

    for class_folder in class_folders:
        class_path = os.path.join(image_directory, class_folder)
        image_files = get_image_files(class_path)

        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            image = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = transform(image).unsqueeze(0)
            

            model = torch.jit.load(model_path)

            with torch.no_grad():
                model.eval()
                output = model(input_tensor)
                output.shape
                idx = F.softmax(output, dim=1).topk(1).indices.squeeze()
                predicted_label = labels[idx] 
                print(predicted_label)
                predicted_labels.append(output)
                
            #print(predicted_labels.shape)
            true_labels=[285,285,285,281,207,158,167,167,235] #Note: this is dummy model with a dummy dataset whose labels do not match-test split

    numeric_predicted_labels = np.array(predicted_labels).squeeze()
    print(numeric_predicted_labels)
    

    true_labels = torch.tensor(true_labels)
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
    
    imagenet_labels_url = "imagenet-simple-labels.json"


    labels = load_labels(imagenet_labels_url)#from actiontracker
    torchscript_model_path ='pretrained_resnet18_scripted.pt' #from s3
    


    for split in splits:
        split_path = os.path.join(r"workspace\6593b7ae10edfa25bc17cfe7-v1.0-imagenet\images", split)
        print(split_path)
        if not os.path.exists(split_path):
            print(f"Warning: Split {split} does not exist.")
            continue

        evaluate_images(torchscript_model_path,labels,split_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Torchscript Evaluation Script')
    parser.add_argument('splits', metavar='SPLIT', nargs='+',
                            help='List of splits to evaluate')

    args = parser.parse_args()
    main(args.splits)