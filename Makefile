# ============================================================
# Brain Tumor MLOps - Makefile
# ============================================================

.PHONY: install test lint train serve docker-build docker-up clean

install:
	pip install -r requirements.txt

lint:
	flake8 src/ tests/ --max-line-length=120

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

train-classifier:
	python src/training/train_classifier.py

train-segmentor:
	python src/training/train_segmentation.py

train-all:
	python pipelines/full_pipeline.py

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

gradio:
	python src/serving/gradio_app.py

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

docker-build:
	docker build -t brain-tumor-api:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .coverage reports/
