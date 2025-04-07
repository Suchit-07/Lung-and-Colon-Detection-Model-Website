import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from .model import load_model
import io

# Initialize FastAPI
app = FastAPI()
#copy file path of attached pth file
#to run, type uvicorn backend.main:app --reload on console
#then go to 127.0.0.1:8000/docs, click try it out, and then put the image file in to get classification and confidence score
MODEL_PATH = 'PUT MODEL PATH HERE'
model = load_model(MODEL_PATH)

# Define image transformations (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to (224, 224), or whatever size your model expects
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize if needed (for pretrained models)
])


# API endpoint to process image
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):#file: UploadFile = File(...)):
    try:
        
        # Preprocess
        #image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted_class].item() * 100  # convert to %

        cancers = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']
        


        return {"prediction": cancers[predicted_class], "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    # Safe to run this only when script is executed directly
    import requests

    url = "http://127.0.0.1:8000/predict"
    files = {'file': open("test_image.jpg", "rb")}
    response = requests.post(url, files=files)
    print(response.json())
