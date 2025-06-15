#!/bin/bash
#SBATCH --job-name=inf-qwen32
#SBATCH --output=qwen32-inf--%j.out
#SBATCH --error=qwen32-inf-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:2
#SBATCH --nodes=1
#SBATCH --mem=80G

module load CUDA/12.1.1

mkdir -p $TMPDIR/cg/merged_model

export HF_HOME=/scratch/$USER/huggingface_cache

source $HOME/.bashrc
source .qwen_unsloth_env/bin/activate

huggingface-cli login --token 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

python test_quantize_4.py

rm -rf $TMPDIR/cg/merged_model
mkdir -p $TMPDIR/cg/merged_model

python test_quantize_2.py

