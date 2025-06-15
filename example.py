# Reference: https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-llms-in-2024-with-trl.ipynb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments

from datasets import load_dataset


# Load dataset
dataset = load_dataset("json", data_files="dataset/train_data.json", split="train", num_proc=4)
print(f"Dataset size: {len(dataset)} -------------------------------")

# Hugging Face model id
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
new_model = "fine-tune_model"

# BitsAndBytesConfig int-4 config, quantize the model to 4bit.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# # Set chat template to OAI chatML, remove if you start from a fine-tuned model
# model, tokenizer = setup_chat_format(model, tokenizer)


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
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=1,                        # log every 10 steps
    logging_strategy="steps",
    weight_decay=0.001,
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    fp16=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="cosine",             # use constant learning rate scheduler
    push_to_hub=False,                      # push model to hub
)

max_seq_length = 3000 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)

# Start training, the model will be automatically saved to the hub and the output directory
trainer.train(resume_from_checkpoint=False)

# Save model 
trainer.save_model()