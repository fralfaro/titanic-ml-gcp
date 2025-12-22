#!/usr/bin/env bash
set -e

echo "🚀 Inicializando estructura del proyecto..."

mkdir -p src/titanic_ml/{data,training,evaluation}
mkdir -p tests notebooks docker scripts .github/workflows

touch pyproject.toml
touch .ruff.toml

touch src/titanic_ml/__init__.py
touch src/titanic_ml/config.py
touch src/titanic_ml/data/preprocess.py
touch src/titanic_ml/training/train.py
touch src/titanic_ml/evaluation/evaluate.py

touch tests/test_preprocess.py
touch docker/Dockerfile
touch .github/workflows/ci.yml

echo "✅ Estructura creada correctamente."
