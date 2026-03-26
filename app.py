from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import io
import base64
import os
import csv

# ── Model ─────────────────────────────────────────────────────────────────────
class SkinTypeResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(SkinTypeResNet, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
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

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="DermaScan API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "models/best_model.pth"
PRODUCTS_CSV = "data/products.csv"
CLASS_NAMES  = ["Dry", "Normal", "Oily"]
IMG_SIZE     = 224
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SKIN_INFO = {
    "Oily":   {"emoji": "💧", "color": "#4CAF50", "desc": "Excess sebum production. Pores appear enlarged. Prone to acne and shine."},
    "Dry":    {"emoji": "🍂", "color": "#FF9800", "desc": "Lacks moisture and natural oils. May feel tight or flaky."},
    "Normal": {"emoji": "✨", "color": "#2196F3", "desc": "Well-balanced skin. Neither too oily nor too dry."},
}

BUDGET_TIERS = {
    "all":     (0,      999999),
    "budget":  (0,      499),
    "mid":     (500,    1499),
    "premium": (1500,   999999),
}

# ── Products ──────────────────────────────────────────────────────────────────
def load_products():
    products = []
    if not os.path.exists(PRODUCTS_CSV):
        print(f"Warning: {PRODUCTS_CSV} not found")
        return products
    with open(PRODUCTS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                products.append({
                    "product_name": row["product_name"].strip(),
                    "category":     row["category"].strip(),
                    "skin_type":    row["skin_type"].strip(),
                    "price_inr":    int(row["price_inr"].strip()),
                    "url":          row["url"].strip(),
                    "image_local":  row.get("image_local", "").strip(),
                })
            except Exception as e:
                print(f"Skipping row: {e}")
    print(f"Loaded {len(products)} products")
    return products

PRODUCTS = load_products()

def get_recommendations(skin_type: str, budget: str = "all"):
    lo, hi = BUDGET_TIERS.get(budget, (0, 999999))
    filtered = [
        p for p in PRODUCTS
        if p["skin_type"].lower() == skin_type.lower()
        and lo <= p["price_inr"] <= hi
    ]
    grouped = {}
    for p in filtered:
        grouped.setdefault(p["category"], []).append(p)
    for cat in grouped:
        grouped[cat].sort(key=lambda x: x["price_inr"])
    return grouped

# ── ML model ──────────────────────────────────────────────────────────────────
def load_model():
    m = SkinTypeResNet(num_classes=len(CLASS_NAMES))
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        m.load_state_dict(sd)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}, using random weights")
    m.to(DEVICE); m.eval()
    return m

model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict_image(pil_img):
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    confs = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))}
    return CLASS_NAMES[pred_idx], confs

# ── Camera ────────────────────────────────────────────────────────────────────
camera = None
camera_index = 1

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

# ── Routes ────────────────────────────────────────────────────────────────────
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
    cam = get_camera()
    ret, frame = cam.read()
    if not ret:
        return JSONResponse({"error": "Could not read from camera"}, status_code=500)
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    label, confidences = predict_image(pil_img)
    _, buf = cv2.imencode(".jpg", frame)
    return {
        "prediction": label, "confidences": confidences,
        "info": SKIN_INFO[label],
        "image_b64": base64.b64encode(buf).decode(),
    }

@app.get("/recommend")
async def recommend(
    skin_type: str = Query(...),
    budget:    str = Query("all"),
):
    if skin_type not in CLASS_NAMES:
        return JSONResponse({"error": f"Invalid skin_type. Use one of {CLASS_NAMES}"}, status_code=400)
    grouped = get_recommendations(skin_type, budget)
    return {
        "skin_type": skin_type,
        "budget":    budget,
        "total":     sum(len(v) for v in grouped.values()),
        "products":  grouped,
    }

def gen_frames():
    cam = get_camera()
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        cx, cy, r = w//2, h//2, min(w,h)//4
        cv2.circle(frame, (cx,cy), r, (0,220,180), 2)
        cv2.line(frame, (cx-r-10,cy), (cx+r+10,cy), (0,220,180), 1)
        cv2.line(frame, (cx,cy-r-10), (cx,cy+r+10), (0,220,180), 1)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.get("/camera/stream")
async def camera_stream():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/camera/status")
async def camera_status():
    return {"available": get_camera().isOpened()}

@app.on_event("shutdown")
def shutdown():
    global camera
    if camera: camera.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)