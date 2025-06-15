from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
import torch
from huggingface_hub import notebook_login

# Step 1: Login to Hugging Face Hub
notebook_login()  # Or run `huggingface-cli login --token xxxxx` in terminal

# Step 2: Define model names
base_model_name = "meta-llama/Llama-3.3-70B-Instruct"  # Your base model
lora_adapter_path = "livmonne/fine-tune_model-chatml_full-data_with_special_token"  # Path to LoRA weights
merged_model_name = "llam3.1-70b-exercise-gen-fullset-q4"  # New name for merged model

# Step 3: Load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading base model to 4-bit...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True  # Needed for some models
)

# Step 4: Load LoRA adapter and merge
print("Merging base model and adapter...")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model = model.merge_and_unload()  # Merge LoRA into base model

# Step 5: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# # Step 6: Save merged model locally (optional)
# model.save_pretrained("merged_model")
# tokenizer.save_pretrained("merged_model")

# Step 7: Upload to Hugging Face Hub
print("Uploading to hub...")
model.push_to_hub(merged_model_name)
tokenizer.push_to_hub(merged_model_name)

print(f"Merged model uploaded to: https://huggingface.co/{merged_model_name}")