#!/bin/bash
#SBATCH --job-name=fine-tune-qwen14
#SBATCH --output=sft-qwen14-%j.out
#SBATCH --error=sft-qwen14-%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=40G

module load CUDA/12.1.1

export HF_HOME=/scratch/$USER/huggingface_cache

source $HOME/.bashrc
source .qwen_unsloth_env/bin/activate

huggingface-cli login --token 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
python fine_tune_unsloth.py

