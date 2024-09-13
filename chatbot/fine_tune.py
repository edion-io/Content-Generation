"""
Fine tune Llama3.1 8B Instruct to create a custom exercise chatbot.
"""
# Copyright (C) 2024  Edion Management Systems
import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model


# Initial global variables
EVAL = False
torch_dtype = torch.float16
attn_implementation = "eager"
base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
new_model = "llama3.1-8b-chat-exercise"


def get_model_and_tokenizer(model_id):
    """
    Get model and tokenizer for training
    Args:
        model_id: model id in huggingface
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch_dtype, bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto", attn_implementation=attn_implementation
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# Get base model and its tokenizer
model, tokenizer = get_model_and_tokenizer(base_model)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)

# Process and load dataset
def map_keys_and_values(item):
    return {
        "role": item["from"].replace("human", "user").replace("gpt", "assistant"),
        "content": item["value"]
    }

def apply_template(examples):
    messages = examples["conversations"]
    messages = [map_keys_and_values(message) for message in messages]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = load_dataset("json", data_files="batch_request/all_synthetic_conversations.json", split="train", num_proc=4)
# dataset = dataset.map(apply_template, num_proc=4) # If the data is not applied template yet
print(f"Dataset size: {len(dataset)} -------------------------------")
# print(dataset['text'][3]) # Check if dataset sample is in correct format


# Set training arguments and configure trainer
if EVAL:
    dataset = dataset.train_test_split(test_size=0.1)

    training_arguments = TrainingArguments(
        output_dir=new_model,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=0.2,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=0.1,
        weight_decay=0.01,
        logging_steps=1,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        group_by_length=True,
        push_to_hub=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        max_seq_length=3000,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing= False,
    )
else:
    training_arguments = TrainingArguments(
        output_dir=new_model,
        per_device_train_batch_size=1, # Keep it small to save GPU memory allocation
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=3,
        # weight_decay=0.01,
        logging_steps=1,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        group_by_length=True,
        push_to_hub=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=3000,  # Keep it small to save GPU memory allocation. Otherwise, use more GPUs more expand GPU memories
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing= False,
    )

# Start training
trainer.train()

# Save and push the adapter to huggingface
trainer.model.save_pretrained(new_model)
trainer.model.push_to_hub(new_model, use_temp_dir=False)