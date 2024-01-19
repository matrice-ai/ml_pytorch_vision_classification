from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
import numpy as np
from python_common.services.s3 import download_from_s3
import torch.nn.functional as F

def load_model(model_id):
    
    try:
        download_from_s3(f'{model_id}/model_best.pt','matrice.dev.models','model.pt')
        model = torch.load('model.pt', map_location='cpu')
        
        if str(type(model)) == "<class 'torch.nn.parallel.data_parallel.DataParallel'>":
            model=torch.load('model.pt', map_location='cpu').module
        
    except:
        print("not able to load model weights will use pretrained resnet18")
        model=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        
    if torch.cuda.is_available():
        model.cuda(0)
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
            input_tensor.cuda(0)
            
        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)

    
        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=1)
    
        # Get the confidence for the predicted class
        predicted_class = torch.argmax(output, 1).item()
        confidence = round(probabilities[0, predicted_class].item(), 2)

        return {"category": str(predicted_class), "confidence": confidence}
