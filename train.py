# Copyright (C) 2025  Edion Management Systems
from transformers import SFTTrainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Convert tokens back to text
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    
    # Calculate BLEU as an example metric
    bleu_score = bleu.compute(predictions=predictions, references=[[label] for label in labels])["bleu"]
    return {"bleu": bleu_score}