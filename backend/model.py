import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define the Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(AttentionLayer, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        attention = torch.bmm(query, key)
        attention = nn.functional.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return out + x


class SimpleResNet18(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleResNet18, self).__init__()
        # Original ResNet18 layers
        self.model = models.resnet18(weights=None)  # No pretrained weights
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Remove the default fully connected layer

        # Add custom fully connected layers
        self.fc1 = nn.Linear(in_features, 512)  # Example of a new fully connected layer
        self.fc2 = nn.Linear(512, num_classes)  # Output layer

    def forward(self, x):
        x = self.model(x)  # ResNet18 layers
        x = torch.relu(self.fc1(x))  # Apply ReLU to fc1 output
        x = self.fc2(x)  # Final output
        return x



# Function to load model
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleResNet18(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    return model


# Function for image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


# Prediction function
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, dim=0)
        return predicted_class.item(), confidence.item()


# Example usage
if __name__ == "__main__":
    model_path = r"C:\Users\bappa\Downloads\Lung-and-Colon-Detection-Model-Website\backend\cancer_detection_model (1).pth"  # Replace with your actual model path
    image_path = r"C:\Users\bappa\Downloads\Lung-and-Colon-Detection-Model-Website\backend\test_lung_cancer_file.jpeg"  # Replace with your image path

    model = load_model(model_path)
    image_tensor = preprocess_image(image_path)
    predicted_class, confidence = predict(model, image_tensor)

    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.4f}")
