import sys
import subprocess
from matrice.deploy import MatriceDeploy
from matrice.actionTracker import ActionTracker

from export_formats.openvino.predict import load_model as load_openvino, predict as predict_openvino
from export_formats.torchscript.predict import load_model as load_torchscript, predict as predict_torchscript
from export_formats.onnx.predict import load_model as load_onnx, predict as predict_onnx
from predict import load_model as load_pytorch, predict as predict_pytorch
from eval_utils import setup_tensorrt

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
            setup_tensorrt()
            from export_formats.tensorrt.predict import load_model as load_tensorrt, predict as predict_tensorrt
            model=load_tensorrt(actionTracker)
        elif "openvino" in runtime_framework:
            model=load_openvino(actionTracker)
        print(runtime_framework)
    except Exception as e:
        model=load_pytorch(actionTracker)
        print(f"ERROR: {e}")
        actionTracker.update_status('MDL_LOAD_ERR', 'ERROR', f'Error in model loading: {str(e)}')
        sys.exit(1)
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
        actionTracker.update_status('MDL_PREDICT', 'ERROR', f'Error during prediction: {str(e)}')
        predictions=predict_pytorch(model,image_bytes) # To make pytorch prediction not effected even new update make errors
        sys.exit(1)
    return predictions


def main(action_id, port):
    global actionTracker
    #Getting the actionTracker
    try:
        actionTracker = ActionTracker(action_id)
    except Exception as e:
        actionTracker.log_error(__file__, 'ml_pytorch_vision_classification/main', f'Error Initializing Action Tracker: {str(e)}')
        print(f"Error initializing ActionTracker: {str(e)}")
        sys.exit(1)

    #Deploying the model
    
    try:
        #load_model, predict ,action_id, port
        x = MatriceDeploy(load_model, predict, action_id, port)
        x.start_server()
        actionTracker.update_status('MDL_DPY_ACK', 'OK', 'Model Deployment has been acknowledged')
    except Exception as e:
        actionTracker.update_status('MDL_DPY_ERR', 'ERROR', 'Error in model deployment : ' + str(e))
        sys.exit(1)
    return

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 deploy.py <action_status_id> <port>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
