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
from skimage.transform import resize
import io
from io import BytesIO

def load_model(actionTracker):
    actionTracker.download_model("model.onnx",model_type="exported")
    return onnxruntime.InferenceSession("model.onnx", None)

def predict(session, image_data):
    # Convert byte string to PIL Image
    image = Image.open(BytesIO(image_data)).convert('RGB')

    # Continue with the rest of your processing...
    input_name = session.get_inputs()[0].name

    # Resize the image to the model's input shape
    img = image.resize((224, 224))
    img_data_resized = np.array(img).transpose(2, 0, 1)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    # Normalize the image
    norm_img_data = (img_data_resized / 255 - mean_vec[:, None, None]) / stddev_vec[:, None, None]
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')

    start_time = time.time()
    raw_result = session.run([], {input_name: norm_img_data})
    end_time = time.time()
    inference_time = np.round((end_time - start_time) * 1000, 2)

    def softmax(x):
        x = x.reshape(-1)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    result = softmax(np.array(raw_result)).tolist()
    # Get the index with maximum probability as the predicted class
    predicted_class = np.argmax(result)

    print('Inference time: ' + str(inference_time) + " ms")
    print('Final top prediction is: ' + str(predicted_class))

    return {"category": str(predicted_class), "confidence": round(result[predicted_class], 2)}