# Makefile for Neural Operator Continuous Time Consistency Model

PYTHON := python3
PIP := pip3
PROJECT := neural-consistency-model

# Directories
SRC_DIR := src
SCRIPTS_DIR := scripts
CONFIG_DIR := configs
DATA_DIR := data
CHECKPOINT_DIR := checkpoints

# Default values
BATCH_SIZE := 128
NUM_GPUS := 1
SEED := 42

.PHONY: help install install-dev clean test format lint train-scratch train-distill evaluate generate

help:
	@echo "Neural Operator Continuous Time Consistency Model"
	@echo "================================================"
	@echo "Available commands:"
	@echo "  make install          Install package and dependencies"
	@echo "  make install-dev      Install with development dependencies"
	@echo "  make clean            Clean generated files and caches"
	@echo "  make test             Run tests"
	@echo "  make format           Format code with black"
	@echo "  make lint             Run linting checks"
	@echo "  make train-scratch    Train model from scratch"
	@echo "  make train-distill    Train via distillation"
	@echo "  make evaluate         Evaluate trained model"
	@echo "  make generate         Generate samples"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist
	rm -rf .pytest_cache .coverage htmlcov

test:
	pytest tests/ -v --cov=$(SRC_DIR) --cov-report=html

format:
	black $(SRC_DIR) $(SCRIPTS_DIR) tests/
	isort $(SRC_DIR) $(SCRIPTS_DIR) tests/

lint:
	flake8 $(SRC_DIR) $(SCRIPTS_DIR)
	mypy $(SRC_DIR) --ignore-missing-imports
	black --check $(SRC_DIR) $(SCRIPTS_DIR)
	isort --check-only $(SRC_DIR) $(SCRIPTS_DIR)

# Training commands
train-scratch:
	$(PYTHON) $(SCRIPTS_DIR)/train_consistency.py \
		--config $(CONFIG_DIR)/cifar10_training.yaml \
		--num_gpus $(NUM_GPUS) \
		--seed $(SEED)

train-distill:
	@if [ -z "$(TEACHER)" ]; then \
		echo "Error: TEACHER variable not set. Usage: make train-distill TEACHER=path/to/model.pt"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/train_consistency.py \
		--config $(CONFIG_DIR)/cifar10_distillation.yaml \
		--teacher_checkpoint $(TEACHER) \
		--num_gpus $(NUM_GPUS) \
		--seed $(SEED)

# Evaluation commands
evaluate:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: CHECKPOINT variable not set. Usage: make evaluate CHECKPOINT=path/to/model.pt"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/evaluate.py \
		--checkpoint $(CHECKPOINT) \
		--num_samples 50000 \
		--batch_size 256 \
		--metrics fid is precision recall

generate:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: CHECKPOINT variable not set. Usage: make generate CHECKPOINT=path/to/model.pt"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/generate_samples.py \
		--checkpoint $(CHECKPOINT) \
		--num_samples 64 \
		--num_steps 1 \
		--save_grid

# Data preparation
prepare-data:
	@mkdir -p $(DATA_DIR)
	$(PYTHON) -c "import torchvision; torchvision.datasets.CIFAR10('./$(DATA_DIR)', download=True)"

# Download pre-trained models
download-models:
	@echo "Downloading pre-trained models..."
	@mkdir -p $(CHECKPOINT_DIR)
	# Add wget/curl commands for model downloads when available

# Quick tests
quick-test:
	$(PYTHON) $(SCRIPTS_DIR)/train_consistency.py \
		--config $(CONFIG_DIR)/cifar10_training.yaml \
		--batch_size 32 \
		--iterations 100 \
		--debug

# Docker commands (if using Docker)
docker-build:
	docker build -t $(PROJECT) -f Dockerfile .

docker-run:
	docker run --gpus all -it --rm \
		-v $(PWD):/workspace \
		-v $(PWD)/$(DATA_DIR):/data \
		$(PROJECT) bash

# Tensorboard
tensorboard:
	tensorboard --logdir=$(CHECKPOINT_DIR)/logs --port=6006

# WandB
wandb-login:
	wandb login

# Create experiment directory
new-experiment:
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME variable not set. Usage: make new-experiment NAME=exp_name"; \
		exit 1; \
	fi
	@mkdir -p experiments/$(NAME)
	@cp $(CONFIG_DIR)/cifar10_ctcm.yaml experiments/$(NAME)/config.yaml
	@echo "Created experiment directory: experiments/$(NAME)"

# Profile training
profile:
	$(PYTHON) -m torch.utils.bottleneck $(SCRIPTS_DIR)/train_consistency.py \
		--config $(CONFIG_DIR)/cifar10_training.yaml \
		--iterations 10