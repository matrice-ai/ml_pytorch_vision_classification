import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
import torch.nn.functional as F

def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e., pinned memory)
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_tensor_shape("images")), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_tensor_shape("output")), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    return (h_input, d_input, h_output, d_output, stream)

def load_model(actionTracker):
    
    actionTracker.download_model("model.engine",model_type="exported")
    # Create a TensorRT runtime
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    # Load the TensorRT engine
    with open("model.engine", 'rb') as f:
        engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
    if not engine:
      print(f"Failed to load the engine: {engine}")
    # Create a TensorRT context
    context = engine.create_execution_context()
    buffers = allocate_buffers(engine)

    return (engine, context, buffers)

def do_inference(engine, context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference
    context.execute_v2(bindings=[int(d_input), int(d_output)])
    # Transfer predictions back from device
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()

    return h_output

def predict(model, image_bytes):

    engine = model[0]
    context = model[1]
    buffers = model[2]
    h_input, d_input, h_output, d_output, stream = buffers
    # Convert image bytes to PIL Image
    image = Image.open(BytesIO(image_bytes)).convert('RGB')

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).numpy().astype(np.float32)  # Add batch dimension
    h_input = np.ravel(image)

    output = do_inference(engine, context, h_input, d_input, h_output, d_output, stream)
    # Process the results
    probabilities = F.softmax(torch.tensor(output).to("cpu"), dim=0)
    predicted_class = torch.argmax(probabilities).item()
    confidence = round(probabilities[predicted_class].item(), 2)

    return {"category": str(predicted_class), "confidence": confidence}
