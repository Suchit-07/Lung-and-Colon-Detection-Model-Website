import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from .model import load_model
import io
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:8000", "*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#copy file path of attached pth file and put into MODEL_PATH
#first setup the backend, open a new console and type "uvicorn backend.main:app --reload"
#after setting that up, open another new console (you should have two by this point) and type "npm run dev"
#run it on localhost:3000 
#text me if you have issues
MODEL_PATH = r'C:\Users\bappa\Downloads\Lung-and-Colon-Detection-Model-Website\backend\lung_colon_model.pth'
model = load_model(MODEL_PATH)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])



@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
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
