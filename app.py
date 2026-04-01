"""
app.py — Alzheimer's Detection API
Features: Predict, GradCAM, Longitudinal Tracking, PDF Report, Risk Scoring

Run: python app.py
"""

import os, io, json, base64, warnings, datetime
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── config ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_alzheimer_resnet50.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 224
CLASSES    = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

CLASS_META = {
    "MildDemented":     {"label":"Mild Demented",     "color":"#f97316","severity":2,"risk":"MODERATE","description":"Mild dementia indicators detected. Cognitive decline is present but the patient can still function independently with some assistance. Neurological consultation is recommended promptly."},
    "ModerateDemented": {"label":"Moderate Demented", "color":"#ef4444","severity":3,"risk":"HIGH",    "description":"Moderate dementia stage detected. Significant cognitive impairment affecting daily activities. Immediate neurological evaluation and care planning is strongly advised."},
    "NonDemented":      {"label":"Non Demented",      "color":"#22c55e","severity":0,"risk":"LOW",     "description":"No signs of dementia detected. Brain MRI patterns appear within normal range. Continue routine health monitoring and maintain a healthy lifestyle."},
    "VeryMildDemented": {"label":"Very Mild Demented","color":"#eab308","severity":1,"risk":"LOW-MOD", "description":"Very mild cognitive changes detected. Early-stage indicators present. A comprehensive neurological evaluation is recommended for baseline assessment."},
}

RISK_WEIGHTS = {"NonDemented":0,"VeryMildDemented":1,"MildDemented":2,"ModerateDemented":3}

# in-memory longitudinal store  {patient_id: [scan, ...]}
longitudinal_db: dict = {}

# ── model ─────────────────────────────────────────────────────────────────────
class ResNet50_CNN_LSTM(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.resnet50(weights=None)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.cnn_refine = nn.Sequential(
            nn.Conv2d(2048,512,1), nn.BatchNorm2d(512), nn.ReLU(), nn.Dropout2d(0.3),
            nn.Conv2d(512, 128,1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(0.2),
        )
        self.lstm = nn.LSTM(input_size=128*7, hidden_size=256, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128,64),  nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.cnn_refine(x)
        B,C,H,W = x.shape
        x = x.permute(0,2,3,1).reshape(B,H,W*C)
        x,_ = self.lstm(x)
        return self.classifier(x[:,-1,:])

print(f"Loading model on {DEVICE}...")
model = ResNet50_CNN_LSTM(num_classes=len(CLASSES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model ready.")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── GradCAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        # hook on last conv layer of backbone (layer4 = index 7)
        target = list(model.backbone.children())[-1]
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, class_idx):
        # LSTM backward requires training mode
        self.model.train()
        self.model.zero_grad()
        logits = self.model(tensor)
        logits[0, class_idx].backward()
        self.model.eval()
        weights = self.gradients.mean(dim=[2,3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam

gradcam = GradCAM(model)

def overlay_heatmap(original_pil, cam):
    """Blend GradCAM heatmap onto original image, return base64 PNG."""
    img = np.array(original_pil.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img + 0.5 * heatmap).astype(np.uint8)
    pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── risk score ────────────────────────────────────────────────────────────────
def compute_risk_score(probs_dict):
    """0-100 composite risk score weighted by class severity."""
    score = sum(probs_dict[c] * RISK_WEIGHTS[c] for c in CLASSES) / 3.0 * 100
    return round(score, 1)

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "frontend"), static_url_path="")
CORS(app)

@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "frontend", "index.html"))

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","device":str(DEVICE),"classes":CLASSES})

@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error":"No file provided"}), 400
    file = request.files["file"]
    patient_id = request.form.get("patient_id", "default")
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error":f"Invalid image: {e}"}), 400

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    # prediction
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    meta       = CLASS_META[pred_class]
    probs_dict = {c: float(p) for c,p in zip(CLASSES, probs)}
    risk_score = compute_risk_score(probs_dict)

    # GradCAM
    tensor_grad = transform(img).unsqueeze(0).to(DEVICE).requires_grad_(True)
    cam = gradcam.generate(tensor_grad, pred_idx)
    heatmap_b64 = overlay_heatmap(img, cam)

    # thumbnail for longitudinal
    thumb_buf = io.BytesIO()
    img.resize((80,80)).save(thumb_buf, format="JPEG", quality=60)
    thumb_b64 = base64.b64encode(thumb_buf.getvalue()).decode()

    scan_record = {
        "timestamp":      datetime.datetime.now().isoformat(),
        "classification": pred_class,
        "label":          meta["label"],
        "confidence":     round(confidence, 2),
        "risk_score":     risk_score,
        "color":          meta["color"],
        "probs":          {c: round(float(p)*100,2) for c,p in zip(CLASSES,probs)},
        "thumb":          thumb_b64,
    }
    longitudinal_db.setdefault(patient_id, []).append(scan_record)

    return jsonify({
        "classification":    pred_class,
        "label":             meta["label"],
        "confidence":        round(confidence, 2),
        "color":             meta["color"],
        "risk":              meta["risk"],
        "risk_score":        risk_score,
        "description":       meta["description"],
        "all_probabilities": {c: round(float(p)*100,2) for c,p in zip(CLASSES,probs)},
        "heatmap":           heatmap_b64,
    })

@app.route("/api/longitudinal/<patient_id>", methods=["GET"])
def longitudinal(patient_id):
    scans = longitudinal_db.get(patient_id, [])
    return jsonify({"patient_id": patient_id, "scans": scans})

@app.route("/api/longitudinal/<patient_id>", methods=["DELETE"])
def clear_longitudinal(patient_id):
    longitudinal_db.pop(patient_id, None)
    return jsonify({"status": "cleared"})

@app.route("/api/report", methods=["POST"])
def generate_report():
    """Generate a PDF clinical report."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.units import cm
    except ImportError:
        return jsonify({"error": "reportlab not installed. Run: pip install reportlab"}), 500

    data = request.json or {}
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=20, textColor=colors.HexColor("#6366f1"), spaceAfter=4)
    h2_style    = ParagraphStyle("h2",    parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#1f2937"), spaceBefore=12, spaceAfter=4)
    body_style  = ParagraphStyle("body",  parent=styles["Normal"],   fontSize=10, textColor=colors.HexColor("#374151"), leading=16)
    muted_style = ParagraphStyle("muted", parent=styles["Normal"],   fontSize=9,  textColor=colors.HexColor("#6b7280"))

    story.append(Paragraph("NeuroScan AI — Clinical Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%B %d, %Y %H:%M')}", muted_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb"), spaceAfter=12))

    story.append(Paragraph("Patient Information", h2_style))
    patient_data = [
        ["Patient ID",   data.get("patient_id","N/A")],
        ["Scan Date",    data.get("scan_date", datetime.datetime.now().strftime("%Y-%m-%d"))],
        ["Referring",    data.get("referring", "N/A")],
    ]
    t = Table(patient_data, colWidths=[4*cm, 12*cm])
    t.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),10),("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#6b7280")),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    story.append(t)

    story.append(Paragraph("Classification Result", h2_style))
    cls   = data.get("classification","N/A")
    label = data.get("label", cls)
    conf  = data.get("confidence", 0)
    risk  = data.get("risk_score", 0)
    desc  = data.get("description","")
    result_data = [
        ["Classification", label],
        ["Confidence",     f"{conf:.1f}%"],
        ["Risk Score",     f"{risk}/100"],
        ["Risk Level",     data.get("risk","N/A")],
    ]
    t2 = Table(result_data, colWidths=[4*cm, 12*cm])
    t2.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),10),("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#6b7280")),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    story.append(t2)

    story.append(Paragraph("Clinical Findings", h2_style))
    story.append(Paragraph(desc, body_style))

    story.append(Paragraph("Class Probabilities", h2_style))
    probs = data.get("all_probabilities", {})
    prob_rows = [["Class","Probability"]] + [[k, f"{v:.1f}%"] for k,v in probs.items()]
    t3 = Table(prob_rows, colWidths=[8*cm, 8*cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#6366f1")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTSIZE",(0,0),(-1,-1),10),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f9fafb"),colors.white]),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),6),
    ]))
    story.append(t3)

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb")))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("DISCLAIMER: This report is generated by an AI system for research and educational purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified neurologist for clinical decisions.", muted_style))

    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=f"neuroscan_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

if __name__ == "__main__":
    import threading, webbrowser
    print("="*50)
    print("  NeuroScan AI — Alzheimer's Detection API")
    print(f"  Device : {DEVICE}")
    print("  URL    : http://localhost:5000")
    print("="*50)
    threading.Timer(1.2, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
