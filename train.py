# Copyright (C) 2025  Edion Management Systems
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig

from datasets import load_dataset
import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# PARAMETERS
model_id = "meta-llama/Llama-3.3-70B-Instruct"
new_model = "fine-tune_model-auto_format-trainset_v2_with_special_token"
max_seq_length = 4000 # max sequence length for model and packing of the dataset

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
dataset = load_dataset("json", data_files="../datasets/data/instruction_sets/instruct_questions_chatml_with_tag.json", split="train", num_proc=4)

# BitsAndBytesConfig int-4 config, quantize the model to 4bit.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'], # Modules/type of layers to unfreeze
        task_type="CAUSAL_LM", 
)

args = TrainingArguments(
    output_dir=new_model,                   # directory to save and repository id
    num_train_epochs=2,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=1,                        # log every 10 steps
    logging_strategy="steps",
    weight_decay=0.001,
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use fp16 (for V100, otherwise for newer GPUs use bf16)
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="cosine",             # use constant learning rate scheduler
    push_to_hub=True,                      # push model to hub
)

model = prepare_model_for_kbit_training(model)

# Pass the processed dataset to the SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    dataset_kwargs={
        "add_special_tokens": False,  # Custom tokens added in preprocessing
        "append_concat_token": False, # No need for additional separators
    }
)

# Start training, the model will be automatically saved to the hub and the output directory
trainer.train(resume_from_checkpoint=False)

# Save model
trainer.save_model()

#Push to hub
trainer.push_to_hub()
