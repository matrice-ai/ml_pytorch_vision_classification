from pathlib import Path
#import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import ipywidgets as widgets
import urllib.request
from PIL import Image
import numpy as np



def load_model(model_xml_path):

    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    model = core.read_model(model=model_xml_path)
    compiled_model = core.compile_model(model=model, device_name=device.value)
    output_layer = compiled_model.output(0)
    return compiled_model, output_layer


def predict(compiled_model, output_layer,image_path):
    image = Image.open(image_path).convert("RGB")
    input_image = image.resize((224, 224))
    input_image_array = np.array(input_image)

    input_image = np.expand_dims(input_image_array, 0)
    result_infer = compiled_model([input_image])[output_layer]
    result_index = np.argmax(result_infer)
    return result_infer,result_index



def main():
    model_xml_path =r'C:\Users\Asus\Documents\Matrice.ai\eval\artifacts\v3-small_224_1.0_float.xml'
    image_path=''
    compiled_model, output_layer = load_model(model_xml_path)
    result_infer,result_index = predict(compiled_model, output_layer, image_path)

if __name__ == "__main__":
    main()



