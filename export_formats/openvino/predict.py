from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import ipywidgets as widgets
import urllib.request
from PIL import Image
import numpy as np
import zipfile
from io import BytesIO

def load_model(actionTracker):

    actionTracker.download_model("model_openvino.zip",model_type="exported")

    with zipfile.ZipFile("model_openvino.zip", 'r') as zip_ref:
        zip_ref.extractall("/model_openvino")

    model_xml_path="/model_openvino/model.xml"
    core = ov.Core()
    device_name = 'GPU' if 'GPU' in core.available_devices else 'CPU'
    model = core.read_model(model=model_xml_path)
    compiled_model = core.compile_model(model=model, device_name=device_name)
    output_layer = compiled_model.output(0)
    return (compiled_model, output_layer)


def predict(model_openvino ,image_data):

    compiled_model=model_openvino[0]
    output_layer=model_openvino[1]

    image = Image.open(BytesIO(image_data)).convert('RGB')

    # Resize the image to the model's input shape
    img = image.resize((224, 224))
    img_data_resized = np.array(img).transpose(2, 0, 1)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    # Normalize the image
    norm_img_data = (img_data_resized / 255 - mean_vec[:, None, None]) / stddev_vec[:, None, None]
    input_image = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    # input_image = np.expand_dims(input_image_array, 0)
    result_infer = compiled_model([input_image])[output_layer]
    result_index = np.argmax(result_infer)

    def softmax(x):
        x = x.reshape(-1)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    result = softmax(np.array(result_infer)).tolist()


    return {"category": str(result_index), "confidence": round(result[result_index], 2)}

