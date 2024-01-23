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

def download_file(url, filename):
    urllib.request.urlretrieve(url, filename)

def load_model(model_path):
    return onnxruntime.InferenceSession(model_path, None)

def predict(session,image_path,labels_path):

    with open(labels_path) as f:
        data = json.load(f)
    labels=np.asarray(data)
    image = Image.open(image_path)
    image_data = np.array(image).transpose(2, 0, 1)
    input_name = session.get_inputs()[0].name 
    img_data = image_data.astype('float32')
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

    start_time = time.time()
    raw_result = session.run([], {input_name:norm_img_data})
    end_time = time.time()
    inference_time = np.round((end_time - start_time) * 1000, 2)

    def softmax(x):
        x = x.reshape(-1)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    result=softmax(np.array(raw_result)).tolist()

    idx = np.argmax(result)
    print('========================================')
    print('Final top prediction is: ' + labels[idx])
    print('Inference time: ' + str(inference_time) + " ms")
    sort_idx = np.flip(np.squeeze(np.argsort(result)))
    print('============ Top 5 labels are: ============================')
    print(labels[sort_idx[:5]])
    return result
    


def main():


    image_path=r'C:\Users\Asus\Documents\Matrice.ai\eval\dog.jpg'
    labels_path=('imagenet-simple-labels.json')
    onnx_model_path = r'C:\Users\Asus\Documents\Matrice.ai\eval\resnet50v2\resnet50v2\resnet50v2.onnx'
    
    session = load_model(onnx_model_path)
    predict(session,image_path,labels_path)
   
    
if __name__ == "__main__":
    main()

