"""
> https://huggingface.co/unsloth/Qwen2.5-7B-Instruct
> https://colab.research.google.com/drive/1qN1CEalC70EO1wGKhNxs1go1W9So61R5?usp=sharing#scrollTo=2eSvM9zX_2d3
Requirements:

!pip install unsloth

!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
"""

from unsloth import FastLanguageModel
import torch


model_size = "32B"
new_model_name = f"Qwen2.5-{model_size}-Instruct-Exercise-default-bs4-trainset"
test_dataset = True # test on testset or single example
test_size = -1 # default -1 (all)
train_steps = -1 # default -1 (invalid)
train_epochs = 3 # default -1 (invalid)
batch_size = 2

# dataset_path = "data/instruct_chatml_with_tag.json" # Full dataset
dataset_path = "data/instruction_sets/instruct_train_chatml_with_tag.json" # Training set

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


print(f"Training model: {new_model_name}")
print(f"Test dataset: {test_dataset}")
print(f"Test size: {test_size}")
print(f"Train steps: {train_steps}")
print(f"Train epochs: {train_epochs}")
print(f"Batch size: {batch_size}")
print(f"Dataset path: {dataset_path}")
print(f"Max seq length: {max_seq_length}")
print(f"Dtype: {dtype}")
print(f"Load in 4bit: {load_in_4bit}")


# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    # Can select any from the below:
    # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
    # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
    # And also all Instruct versions and Math. Coding verisons!
    model_name = f"unsloth/Qwen2.5-{model_size}-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
)

def formatting_prompts_func(examples):
    messages = examples['messages']
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return { "text" : text, }
pass

from datasets import load_dataset
dataset = load_dataset("json", data_files=dataset_path, split="train", num_proc=4)


dataset = dataset.map(formatting_prompts_func, batched = True,)

print(dataset[5]["messages"])

print(dataset[5]["text"])


# Train #

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = train_epochs, # Set this for 1 full training run.
        max_steps = train_steps,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

# We also use Unsloth's train_on_completions method to only train on the assistant outputs and ignore the loss on the user's inputs.
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

tokenizer.decode(trainer.train_dataset[5]["input_ids"])

space = tokenizer(" ", add_special_tokens = False).input_ids[0]
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])

# GPU Memory before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Training
trainer_stats = trainer.train()

# GPU Memory after training
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Inference #
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference


if not test_dataset:
    messages = [
        {"role": "user", "content": "Give me a true or false statements exercise for a 4th grade student learning mathematics. It should be an exercise about odd and even numbers."},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 512, use_cache = True,
                            temperature = 1.5, min_p = 0.1)
    print(tokenizer.batch_decode(outputs))

else:
    # Test on dataset
    print("Testing model on test dataset...")
    import pandas as pd
    from datasets import load_dataset
    import os
    
    # Load test data from CSV
    test_file = "data/instruction_sets/instruct_test.csv"
    try:
        if test_size != -1:
            test_df = pd.read_csv(test_file)[:test_size]
        else:
            test_df = pd.read_csv(test_file)
        print(f"Loaded test data with {len(test_df)} examples")
    except Exception as e:
        print(f"Error loading test data: {e}")
        raise
    
    # Initialize list to store predictions
    predictions = []
    
    # Process each example in the test dataset
    for idx, row in test_df.iterrows():
        input_text = row['Input']
        
        # Create message format expected by the model
        messages = [{"role": "user", "content": input_text}]
        
        # Format the input for inference with attention mask
        tokenized_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        
        # Explicitly create attention mask (1 for all tokens in the input)
        attention_mask = torch.ones_like(tokenized_inputs)
        
        # Move tensors to cuda
        inputs = tokenized_inputs.to("cuda")
        attention_mask = attention_mask.to("cuda")
        
        # Generate response with attention mask
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.7,
            do_sample=True
        )
        
        # Decode response
        response = tokenizer.batch_decode(outputs)[0]
        
        # Add prediction to list
        predictions.append(response)
        
        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} examples")
    
    # Add predictions to the dataframe
    test_df['prediction'] = predictions
    
    # Save results to a new CSV file
    output_file = f"{new_model_name}_test_result.csv"
    test_df.to_csv(output_file, index=False)
    print(f"Saved test results to {output_file}")

# Save #

model.save_pretrained(new_model_name) # Local saving
tokenizer.save_pretrained(new_model_name)

model.push_to_hub(new_model_name) # Online saving
tokenizer.push_to_hub(new_model_name) # Online saving