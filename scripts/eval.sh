#!/bin/bash
# Eval the model given as argument on all available datasets
set -e
MODEL_ID=$1

uv run scripts/inference.py $MODEL_ID --dataset_name ai2d --save True --overwrite True
uv run scripts/inference.py $MODEL_ID --dataset_name chartqa --save True --overwrite True
uv run scripts/inference.py $MODEL_ID --dataset_name realworldqa --save True --overwrite True
uv run scripts/inference.py $MODEL_ID --dataset_name textvqa --save True --overwrite True
uv run scripts/inference.py $MODEL_ID --dataset_name mme --save True --overwrite True
uv run scripts/inference.py $MODEL_ID --dataset_name ocrbench --save True --overwrite True
uv run scripts/inference.py $MODEL_ID --dataset_name docvqa --save True --overwrite True
uv run scripts/inference.py $MODEL_ID --dataset_name infographic_vqa --save True --overwrite True
