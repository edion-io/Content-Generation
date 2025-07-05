""" Manual Evaluation Script for Exercise Generation

This script is used to generate exercises based on prompts from a CSV file using a quantized Llama-3.3-70B model.

The csv file should contain two columns: 'Input' for the prompts and 'Samples' for the number of samples to generate for each prompt.
The prompts need to be fully formulated instructions, there is no handling of the S T G D M parameters.
It requires a compatible GPU with CUDA support and the necessary libraries installed. (Torch, Transformers, Pandas, BitsAndBytes)

Copyright (C) 2025  Edion Management Systems

"""

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and the necessary drivers installed.")

base_model = "meta-llama/Llama-3.3-70B-Instruct"
quantized_model = "livmonne/llam3.3-70b-exercise-gen-trainset-post-q4"
quantized_model_path = "quantized_model_llama3.3_70B"

if not os.path.exists(quantized_model_path) or not os.listdir(quantized_model_path):
    print("Quantized model not found, downloading...")
    os.makedirs(quantized_model_path, exist_ok=True)
    quantized_model = AutoModelForCausalLM.from_pretrained(quantized_model, device_map="auto", torch_dtype=torch.bfloat16)
    # print("Saving quantized model")
    # quantized_model.save_pretrained(quantized_model_path)
    # del quantized_model
    # torch.cuda.empty_cache()
    # print("Quantized model saved")

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=quantized_model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

system_prompt = """You are a smart and helpful excercise desinger for students from grade level 1-12. 
Follow these guildlines to design excercises:

1. Read and understand the instructions given and make sure your exercise are following them.
2. Use Latex as the format of your response.
3. Limit the length of excercise at 200 tokens, avoid repetition.
4. Be creative, try to increase the variety of your exercises.
5. Make sure your language and sentences can be understood by the corresponding students."""

# Gnerate function (In pipeline)
def generate(prompt, max_new_tokens=512):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    outputs = pipe(messages,
                   max_new_tokens=max_new_tokens,
                   temperature=0.7,
                   top_p=0.9,
                   top_k=50)
    # outputs = pipe(messages,
    #                max_new_tokens=max_new_tokens,
    #                temperature=0.7,
    #                do_sample=True,
    #                top_p=0.9,
    #                repetition_penalty=1.2,
    #                no_repeat_ngram_size=3)
    response = outputs[0]["generated_text"][-1]["content"]
    return response

print("Start generation...")
count = 0
data = pd.read_csv("test_data.csv", names =["Input", "Samples"], header=None, skiprows=1)
if data.empty:
    raise ValueError("The input data is empty. Please provide a valid CSV file with prompts.")

result = []
for prompt, samples in zip(data.Input, data["Samples"]):
    print(f"<<{count} Prompt>>\n", prompt, "\n<<Prompt>>\n\n")
    for _ in range(int(samples)):
        response = generate(prompt, max_new_tokens=100000)
        print(response, "\n")
        result.append({"prompt": prompt, "prediction": response})
    print()
    count += 1
    # break
# pd.DataFrame(result).to_csv("./generation/test_output4.csv", index=False)
