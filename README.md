# NeuroScan AI — Alzheimer's Disease Detection

Deep learning system for Alzheimer's MRI classification using ResNet50 + CNN + LSTM.

## Results
- Val Accuracy: 92%
- Macro F1: 0.94
- Classes: NonDemented, VeryMildDemented, MildDemented, ModerateDemented

## Architecture
ResNet50 (pretrained, layer2-4 unfrozen) → CNN refinement → LSTM → Classifier

## Setup

```bash
pip install -r requirements.txt
```

## Dataset
Download from [Mendeley](https://data.mendeley.com/datasets/xx9zzz6t54) and place the zip in this folder, then:

```bash
python setup_dataset.py
```

## Train

```bash
python run_pipeline.py
```

## Run App

```bash
python app.py
```

Opens browser at `http://localhost:5000` with full UI including:
- MRI upload and classification
- GradCAM attention heatmap
- Risk scoring
- Longitudinal tracking
- PDF report export
