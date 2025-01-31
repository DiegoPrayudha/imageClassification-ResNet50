
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import models, transforms
import io

# Inisialisasi FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World",
            "version": "1.0.0"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet50(weights=None)

model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(256, 120)  # Ganti 120 dengan jumlah kelas sesuai kebutuhan
)

# Load state_dict
checkpoint = torch.load('best_model.pth', map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
class_names = checkpoint['class_names']
model = model.to(device)
model.eval()

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Fungsi untuk prediksi
def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device) 
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)
    predicted_class = class_names[predicted_idx.item()] 
    return predicted_class, confidence.item()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Pastikan file memiliki format yang didukung
        if not file.content_type.startswith("image/"):
            return JSONResponse({"error": "File bukan gambar"}, status_code=400)
        
        # Baca file gambar
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        
        # Prediksi label dan confidence
        predicted_idx, confidence = predict_image(image)
        return JSONResponse({
            "predicted_index": predicted_idx,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse({"error": f"Error: {str(e)}"}, status_code=500)
