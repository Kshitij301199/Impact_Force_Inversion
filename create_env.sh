#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-environment.yml}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: $ENV_FILE not found in $(pwd)."
  exit 1
fi

echo "Creating or updating conda environment from $ENV_FILE..."
if conda env create -f "$ENV_FILE"; then
  echo "Environment created."
else
  echo "Environment may already exist; updating instead..."
  conda env update -f "$ENV_FILE" --prune
fi

ENV_NAME=$(grep '^name:' "$ENV_FILE" | awk '{print $2}' || true)
if [[ -n "$ENV_NAME" ]]; then
  echo "To activate the environment: conda activate $ENV_NAME"
else
  echo "Environment created/updated. Activate it manually."
fi

echo "Done."
