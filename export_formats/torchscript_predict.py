import torch
import torch.nn.functional as F
from torchvision import transforms,models
from PIL import Image
import json

def load_model(model_path):
    model = torch.jit.load(model_path)
    return model

def predict(image_path, model, label_file_path):
    with open(label_file_path, 'r') as f:
        labels = json.load(f)
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)
    top5_indices = F.softmax(output, dim=1).topk(5).indices.squeeze().tolist()
    predicted_labels = [labels[idx] for idx in top5_indices]
    
    return predicted_labels

def main():
    pretrained_model = models.resnet18(pretrained=True)
    pretrained_model_scripted = torch.jit.script(pretrained_model)
    pretrained_model_scripted.save('pretrained_resnet18_scripted.pt')
    label_file_path = r'C:/Users/Asus/Documents/Matrice.ai/eval/imagenet-simple-labels.json'
    model_path = 'pretrained_resnet18_scripted.pt'
    model=load_model(model_path)
    image_path = r'C:\Users\Asus\Documents\Matrice.ai\eval\dog.jpg'  # Replace with the path to your image
    predicted_labels = predict(image_path, model, label_file_path)
    
    print('Predicted Labels:\n', predicted_labels)

if __name__ == "__main__":
    main()

