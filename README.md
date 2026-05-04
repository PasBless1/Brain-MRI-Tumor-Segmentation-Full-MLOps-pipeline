# 🧠 MRI Brain Tumor Segmentation — Full MLOps Pipeline

A production-grade MLOps project for MRI brain tumor detection and segmentation using **ResNet50** (classifier) and **ResUNet** (segmentation) with full experiment tracking, CI/CD, and model serving.

## 🗂️ Project Structure

```
mri-brain-tumor-mlops/
├── configs/config.yaml          # All hyperparameters in one place
├── src/
│   ├── data/                    # Dataset loading, preprocessing, CLAHE
│   ├── models/                  # ResNet50 classifier + ResUNet
│   ├── losses.py                # Focal Tversky loss
│   ├── training/                # MLflow-tracked training scripts
│   ├── evaluation/              # Dice, IoU, AUC, confusion matrix
│   ├── serving/                 # FastAPI REST API + Gradio UI
│   └── monitoring/              # Evidently drift detection
├── pipelines/full_pipeline.py   # End-to-end pipeline runner
├── tests/                       # Unit + integration tests (pytest)
├── .github/workflows/ci_cd.yml  # GitHub Actions CI/CD
├── Dockerfile                   # Container for API
├── docker-compose.yml           # MLflow + API + Gradio
├── Makefile                     # Developer shortcuts
├── MRI_BrainTumor_MLOps_Colab.ipynb  # ← Colab GPU training notebook
└── requirements.txt
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
make install

# 2. Start MLflow tracking server + services
make docker-up

# 3. Run full pipeline (train + evaluate + register)
make train-all

# 4. Serve the API
make serve

# 5. Launch Gradio demo
make gradio
```

## 📊 MLOps Stack

| Component | Tool |
|---|---|
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| API Serving | FastAPI + Uvicorn |
| Demo UI | Gradio |
| Containerization | Docker + Compose |
| CI/CD | GitHub Actions |
| Testing | pytest + pytest-cov |
| Monitoring | Evidently AI |
| Colab Training | Jupyter Notebook (GPU) |
| Config Management | YAML (single source of truth) |

## 📈 Model Performance

| Model | Metric | Score |
|---|---|---|
| ResNet50 Classifier | Accuracy | ~80% |
| ResNet50 Classifier | AUC | ~0.85 |
| ResUNet Segmentor | Dice Score | ~0.92 |
| ResUNet Segmentor | Tversky Score | ~0.98 |
