import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
import torch.nn.functional as F

def load_model(actionTracker):
    
    actionTracker.download_model("model.engine",model_type="exported")
    # Create a TensorRT runtime
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    # Load the TensorRT engine
    with open("model.engine", 'rb') as f:
        engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)

    # Create a TensorRT context
    context = engine.create_execution_context()

    return (engine, context)


def predict(model, image_bytes):

    engine, context = model[0] ,model[1]
    # Convert image bytes to PIL Image
    image = Image.open(BytesIO(image_bytes)).convert('RGB')

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Allocate GPU memory for input and output buffers
    input_size = trt.volume(engine.get_tensor_shape("images"))
    output_size = trt.volume(engine.get_tensor_shape("output"))
    
    input_buffer = cuda.pagelocked_empty(input_size, dtype=np.float32)
    output_buffer = cuda.pagelocked_empty(output_size, dtype=np.float32)
    
    # Copy input data to GPU
    np.copyto(input_buffer, input_tensor.numpy().ravel())
    cuda.memcpy_htod(cuda.mem_alloc(input_buffer.nbytes), input_buffer)
    
    # Run inference
    context.execute_v2(bindings=[cuda.mem_alloc(input_buffer.nbytes), cuda.mem_alloc(output_buffer.nbytes)])
    
    # Copy output data from GPU
    cuda.memcpy_dtoh(output_buffer, cuda.mem_alloc(output_buffer.nbytes))

    # Process the results
    probabilities = F.softmax(torch.from_numpy(output_buffer).to("cpu"), dim=0)
    predicted_class = torch.argmax(probabilities).item()
    confidence = round(probabilities[predicted_class].item(), 2)

    return {"category": str(predicted_class), "confidence": confidence}
