# backend/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .model import load_model, preprocess_image, predict as predict_image
import shutil
import os

app = FastAPI()
os.makedirs("uploads", exist_ok=True)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
MODEL_PATH = r"C:\Users\bappa\Downloads\Lung-and-Colon-Detection-Model-Website\backend\cancer_detection_model (1).pth"  # or the exact name of your file
model = load_model(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Cancer Detection API"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"

    # Save image to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Preprocess & predict
    image_tensor = preprocess_image(file_path)
    predicted_class, confidence = predict_image(model, image_tensor)

    # Clean up
    os.remove(file_path)

    return {
        "class": predicted_class,
        "confidence": confidence
    }
