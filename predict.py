from PIL import Image
from io import BytesIO
import sys
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

def load_model(actionTracker):
    try:
        actionTracker.download_model("model.pt")
        model = torch.load('model.pt', map_location='cpu')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device is :{device}")
        model = model.to(device)
        model.eval()
        actionTracker.update_status('MDL_PRED_STRT', 'OK', 'Model loaded for prediction')
        return model
    
    except Exception as e:
        actionTracker.update_status('MDL_PRED_ERR', 'ERROR', 'Error in loading model')
        actionTracker.log_error(__file__, 'ml_pytorch_vision_classification/main', f'Error updating status to MDL_PRED_STRT: {str(e)}')
        print(f"Error updating status to MDL_PRED_STRT: {str(e)}")
        sys.exit(1)

def predict(model, image_bytes):
    # Convert image bytes to PIL Image
    image = Image.open(BytesIO(image_bytes)).convert('RGB')

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Move input tensor to the same device as model parameters
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1)

    # Get the confidence for the predicted class
    predicted_class = torch.argmax(output, 1).item()
    confidence = round(probabilities[0, predicted_class].item(), 2)

    return {"category": str(predicted_class), "confidence": confidence}
            

