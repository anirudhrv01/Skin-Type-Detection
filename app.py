from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from PIL import Image


class SkinTypeResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(SkinTypeResNet, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Freeze backbone (matches training setup)
        for name, param in self.model.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
import numpy as np
import cv2
import io
import base64
import os

app = FastAPI(title="Skin Type Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = "models/best_model.pth"
CLASS_NAMES = ["Dry", "Normal", "Oily"]
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Skin type descriptions ───────────────────────────────────────────────────
SKIN_INFO = {
    "Oily":   {"emoji": "💧", "color": "#4CAF50", "desc": "Excess sebum production. Pores appear enlarged. Prone to acne and shine."},
    "Dry":    {"emoji": "🍂", "color": "#FF9800", "desc": "Lacks moisture and natural oils. May feel tight or flaky."},
    "Normal": {"emoji": "✨", "color": "#2196F3", "desc": "Well-balanced skin. Neither too oily nor too dry."},
}

# ─── Model ────────────────────────────────────────────────────────────────────
def load_model():
    model = SkinTypeResNet(num_classes=len(CLASS_NAMES))
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        # If saved as plain state_dict (not wrapped in a dict)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️  Model file not found at {MODEL_PATH}. Using random weights for demo.")
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict_image(pil_img: Image.Image):
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    confidences = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))}
    return CLASS_NAMES[pred_idx], confidences

# ─── Camera state ─────────────────────────────────────────────────────────────
camera = None
camera_index = 1   # change to 1,2… if USB microscope is not default cam

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    label, confidences = predict_image(pil_img)
    return {"prediction": label, "confidences": confidences, "info": SKIN_INFO[label]}

@app.post("/predict/capture")
async def predict_capture():
    """Capture a frame from the USB microscope and predict."""
    cam = get_camera()
    ret, frame = cam.read()
    if not ret:
        return JSONResponse({"error": "Could not read from camera"}, status_code=500)
    # Convert BGR→RGB for PIL
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    label, confidences = predict_image(pil_img)

    # Also send back the captured frame as base64 for display
    _, buf = cv2.imencode(".jpg", frame)
    img_b64 = base64.b64encode(buf).decode()

    return {
        "prediction": label,
        "confidences": confidences,
        "info": SKIN_INFO[label],
        "image_b64": img_b64,
    }

def gen_frames():
    cam = get_camera()
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        # Draw crosshair guide
        h, w = frame.shape[:2]
        cx, cy, r = w // 2, h // 2, min(w, h) // 4
        cv2.circle(frame, (cx, cy), r, (0, 220, 180), 2)
        cv2.line(frame, (cx - r - 10, cy), (cx + r + 10, cy), (0, 220, 180), 1)
        cv2.line(frame, (cx, cy - r - 10), (cx, cy + r + 10), (0, 220, 180), 1)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.get("/camera/stream")
async def camera_stream():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/camera/status")
async def camera_status():
    cam = get_camera()
    return {"available": cam.isOpened()}

@app.on_event("shutdown")
def shutdown():
    global camera
    if camera:
        camera.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)