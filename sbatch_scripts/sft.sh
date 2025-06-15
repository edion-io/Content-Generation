#!/bin/bash
#SBATCH --job-name=fine-tune-l70
#SBATCH --output=sft-%j.out
#SBATCH --error=sft-%j.err
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:3
#SBATCH --nodes=1
#SBATCH --mem=200G

module load CUDA/12.1.1

mkdir -p $TMPDIR/cg/data
cp -r data/* $TMPDIR/cg/data
cp train.py $TMPDIR/cg
cp test_quantize.py $TMPDIR/cg

cd $TMPDIR/cg

export HF_HOME=/scratch/$USER/huggingface_cache

source $HOME/.bashrc
conda activate unsloth_env

huggingface-cli login --token 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
python train.py
python test_quantize.py

cp -r $TMPDIR/* /scratch/$USER/
