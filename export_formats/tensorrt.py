import torch
import torch_tensorrt
from torch_tensorrt import Input
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os
import json
import argparse
import torchvision.models as models

def load_and_convert_model(model_path, input_shape, precision=torch.half):
    resnet18 = models.resnet18(pretrained=True).half().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda").half()]
    
    trt_model = torch_tensorrt.compile(
        resnet18,
        inputs=inputs,
        enabled_precisions={precision}
    )

    return trt_model

def inference(trt_model, img_batch, topk=5):
    trt_model.eval()
    with torch.no_grad():
        outputs = trt_model(img_batch)
    prob = torch.nn.functional.softmax(outputs[0], dim=1)

    probs, classes = torch.topk(prob, topk, dim=1)
    return probs, classes

def preprocess_image(img_dir, img_path):
    img_path = os.path.join(img_dir, img_path)
    image = Image.open(img_path).convert("RGB")
    input_image = image.resize((224, 224))
    input_image_array = np.array(input_image)
    img = transforms.ToTensor()(input_image_array).unsqueeze(0).to("cuda")
    return img


def main():
    # Set paths relative to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "resnet18.pt")
    img_dir = os.path.join(current_dir, "images")
    categories_path = os.path.join(current_dir, "imagenet_class_index.json")

    # Load and convert the model
    input_shape = (1, 3, 224, 224)
    trt_model = load_and_convert_model(model_path, input_shape)

    # Load the batch of images
    img_batch = torch.cat([preprocess_image(img_dir, img_path).half() for img_path in os.listdir(img_dir)])

    # Run inference
    topk = 5
    probs, classes = inference(trt_model, img_batch, topk)
    print(probs,classes)
    
    with open(categories_path, 'r') as f:
        categories = json.load(f)

    # Display results
  
   # Display results
    for img_path, prob, class_indices in zip(os.listdir(img_dir), probs, classes):
        print(f"Results for {img_path}:")
        for j in range(topk):
            probability = prob[j].item()
            class_index = int(class_indices[j])
            
            # Get label from 'categories' dictionary
            if str(class_index) in categories:
                class_label = categories[str(class_index)][1]
            else:
                class_label = "Unknown"
            
            print(f"Top {j + 1}: {class_label} - Probability: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
