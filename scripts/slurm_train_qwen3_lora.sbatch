#!/bin/bash
#SBATCH --job-name=legalmind-lora
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/zzhu126/LegalMind-RAG/logs/slurm/%x-%j.out
#SBATCH --error=/data/zzhu126/LegalMind-RAG/logs/slurm/%x-%j.err

set -e

cd /data/zzhu126/LegalMind-RAG
export PYTHONUNBUFFERED=1

/data/zzhu126/environments/legalmind-rag-gpu/bin/python \
  scripts/train_qlora.py \
  --model-config configs/model_qwen3_4b_school_lora.yaml \
  --training-config configs/classification/qwen3_4b_lora_school_full.yaml
