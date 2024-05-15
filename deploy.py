import sys
import subprocess
from python_sdk.src.matrice_sdk.deploy import MatriceDeploy
from python_sdk.src.matrice_sdk.actionTracker import ActionTracker
from python_sdk.src.matrice_sdk.matrice import Session

from export_formats.openvino.predict import load_model as load_openvino, predict as predict_openvino
from export_formats.torchscript.predict import load_model as load_torchscript, predict as predict_torchscript
from export_formats.onnx.predict import load_model as load_onnx, predict as predict_onnx
from predict import load_model as load_pytorch, predict as predict_pytorch

def load_model(actionTracker):
    runtime_framework=actionTracker.action_details['runtimeFramework'].lower()
    try:
        if "onnx" in runtime_framework:
            model=load_onnx(actionTracker)
        elif "torchscript" in runtime_framework:
            model=load_torchscript(actionTracker)
        elif "pytorch" in runtime_framework:
            model=load_pytorch(actionTracker)
        elif "tensorrt" in runtime_framework:
            subprocess.run(["pip", "install", "tensorrt", "pycuda"]) # Very large to add to requirements.txt
            from export_formats.tensorrt.predict import load_model as load_tensorrt, predict as predict_tensorrt
            model=load_tensorrt(actionTracker)
        elif "openvino" in runtime_framework:
            model=load_openvino(actionTracker)
        print(runtime_framework)
    except Exception as e:
        print(f"ERROR: {e}")
        model=load_pytorch(actionTracker)
    model_data={
        "framework": runtime_framework,
        "model": model
        }
    
    return model_data

def predict(model_data,image_bytes):
    runtime_framework=model_data['framework']
    model=model_data['model']
    try:
        if "onnx" in runtime_framework:
            predictions=predict_onnx(model,image_bytes)
        elif "torchscript" in runtime_framework:
            predictions=predict_torchscript(model,image_bytes)
        elif "pytorch" in runtime_framework:
            predictions=predict_pytorch(model,image_bytes)
        elif "tensorrt" in runtime_framework:
            predictions=predict_tensorrt(model,image_bytes)
        elif "openvino" in runtime_framework:
            predictions=predict_openvino(model,image_bytes)
    except Exception as e:
        print(f"ERROR: {e}")
        predictions=predict_pytorch(model,image_bytes) # To make pytorch prediction not effected even new update make errors
    return predictions

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 deploy.py <action_status_id> <port>")
        sys.exit(1)
    
    action_status_id = sys.argv[1]
    port=sys.argv[2]
    session=Session()
    x=MatriceDeploy(session,load_model, predict ,action_status_id,port)
    x.start_server()