from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
import numpy as np

import torch.nn.functional as F

def load_model(actionTracker):
    

    actionTracker.download_model("model.pt")
    model = torch.load('model.pt', map_location='cpu')

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        
    model.eval()
    return model

def predict(model,image_bytes):
        # Convert image bytes to PIL Image
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            input_tensor.to(device)
            
        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)

    
        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=1)
    
        # Get the confidence for the predicted class
        predicted_class = torch.argmax(output, 1).item()
        confidence = round(probabilities[0, predicted_class].item(), 2)

        return {"category": str(predicted_class), "confidence": confidence}
