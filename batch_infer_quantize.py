"""
A general script to load model, quantize model, merge model and inference in batch.

Use cases:
    1. Load a full precision or quantized model and do batch inference.
        ```
        MODEL_NAME = "unsloth/Qwen2.5-32B-Instruct" 
        LORA_ADPATER = None
        DATASET_PATH = "data/instruct_test.csv"
        BATCH_SIZE = 4
        MAX_NEW_TOKENS = 256
        TEMPERATURE = 0.7 
        QUANTIZE = False
        UPLOAD = False
        UPLOAD_NAME = None
        ```
    2. Load lora adapter and merge to base model, then do batch inference.
        ```
        MODEL_NAME = "unsloth/Qwen2.5-32B-Instruct" 
        LORA_ADPATER = "livmonne/Qwen2.5-32B-Instruct-Exercise-default-bs4-trainset"
        DATASET_PATH = "data/instruct_test.csv"
        BATCH_SIZE = 4
        MAX_NEW_TOKENS = 256
        TEMPERATURE = 0.7 
        QUANTIZE = False
        UPLOAD = False
        UPLOAD_NAME = None
        ```
    3. Load lora adapter and merge to base model, quantize it to 4 bit, upload to huggingface, then do batch inference.
        ```
        MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct" 
        LORA_ADPATER = "livmonne/fine-tune_model-chatml_full-data_with_special_token"
        DATASET_PATH = "data/instruct_test.csv"
        BATCH_SIZE = 4
        MAX_NEW_TOKENS = 256
        TEMPERATURE = 0.7 
        QUANTIZE = True
        UPLOAD = True
        UPLOAD_NAME = "llam3.3-70b-exercise-gen-fullset-q4"
        ```
"""

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pandas as pd

# Configuration
MODEL_NAME = "unsloth/Qwen2.5-32B-Instruct"    # Base model name path/repos
LORA_ADPATER = "livmonne/Qwen2.5-32B-Instruct-Exercise-default-bs4-trainset"    # Lora Adapter path/repos
DATASET_PATH = "data/instruct_test.csv"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7 
QUANTIZE = False   # Weather to quantize model
UPLOAD = False  # Weather to upload merged model
# UPLOAD_NAME = "Qwen2.5-32B-exercise-gen-fullset-q4"    # Merged model name to be uploaded
UPLOAD_NAME = None

SYSTEM_PROMPT = """You are a smart and helpful exercise designer for students from grade level 1-12. 
Follow these guidelines to design exercises:

1. Read and understand the instructions given and make sure your exercises are following them.
2. Use LaTeX as the format of your response.
3. Limit the length of the exercise to 200 tokens, avoid repetition.
4. Be creative, try to increase the variety of your exercises.
5. Make sure your language and sentences can be understood by the corresponding students."""


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading full precision model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

if LORA_ADPATER:
    from peft import PeftModel
    print("Merging lora adapter to base model...")
    model = PeftModel.from_pretrained(model, LORA_ADPATER)
    model = model.merge_and_unload()  # Merge LoRA into base model
    model.save_pretrained("/tmp/cg/merged_model")
    print("Saved merged model")
    del model
    torch.cuda.empty_cache()

if QUANTIZE:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print("Loading quantized model...")
    model = AutoModelForCausalLM.from_pretrained(
        "/tmp/cg/merged_model",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True  # Needed for some models
    )

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Load data
print("Loading dataset...")
inputs = pd.read_csv(DATASET_PATH)["Input"].tolist()

# Prepare prompts with system message
def create_chat_prompt(user_message):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user_message}],
        add_generation_prompt=True,
        tokenize=False
    )

formatted_inputs = [create_chat_prompt(text) for text in inputs]

# Generate outputs in batches
generations = []
for i in tqdm(range(0, len(formatted_inputs), BATCH_SIZE), desc="Generating"):
    batch_prompts = formatted_inputs[i:i+BATCH_SIZE]
    output_batch = generator(
        batch_prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.9,
        top_k=50
    )
    # Extract only the generated part (after the prompt)
    generations.extend([
        result[0]["generated_text"][len(prompt):].strip()
        for result, prompt in zip(output_batch, batch_prompts)
    ])

# Save the output
output_df = pd.DataFrame({
    "Input": inputs,
    "Generated": generations
})
output_df.to_csv("test_output_llama70b-q4.csv", index=False)
print("Saved output to output_with_generations_pipeline.csv")


# --- Add GPU memory reporting ---
import gc

# Force garbage collection and synchronize
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

# Total max reserved memory across all GPUs (in GB)
total_reserved = 0
for i in range(torch.cuda.device_count()):
    reserved = torch.cuda.max_memory_reserved(i) / (1024 ** 3)  # Convert to GB
    print(f"Max reserved memory on GPU {i}: {reserved:.2f} GB")
    total_reserved += reserved

print(f"\nTotal max reserved memory across GPUs: {total_reserved:.2f} GB")