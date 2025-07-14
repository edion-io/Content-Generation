import torch
import pandas as pd
from transformers import pipeline
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


base_model = "meta-llama/Llama-3.3-70B-Instruct"
# adapter = "livmonne/fine-tune_model"
# adapter = "/scratch/s4702328/cg/fine-tune_model/checkpoint-2078"
adapter = "livmonne/fine-tune_model-auto_format-trainset_v2_with_special_token"

# Reload base model and its tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
)

# Merge (fine tuned) adapter with base model
model = PeftModel.from_pretrained(base_model_reload, adapter)
model = model.merge_and_unload()

# merged_model = "/scratch/s4702328/merged_model_8b_llama"
merged_model = "merged_model_llama3.3_70B"

print("Saving merged model")
model.save_pretrained(merged_model)
del model
torch.cuda.empty_cache() 
print("Merged model saved")

# Reload and test
# merged_model = "/scratch/s4702328/merged_model_llama3.3_70B"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

print("Reloading model and quantizing to 4bit")
quantized_model = AutoModelForCausalLM.from_pretrained(
    merged_model, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)

print("Saving quantized model")
# quantized_model_path = "/scratch/s4702328/quantized_model_8b_llama"
quantized_model_path = "quantized_model_llama3.3_70B"
quantized_model.save_pretrained(quantized_model_path)
del quantized_model
torch.cuda.empty_cache()
print("Quantized model saved")

print("Loading final testing model")
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
print("Testing model loaded")

# Set pipeline for LLM task
pipe = pipeline(
    "text-generation",
    model=quantized_model_path,
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
data = pd.read_csv("data/instruction_sets/instruct_test.csv")
result = []
for prompt in data.Input:
    print(f"<<{count} Prompt>>", prompt, "<<Prompt>>") 
    response = generate(prompt, max_new_tokens=300)
    print(f"<<{count} Generation>>", response, "<<Generation>>")
    result.append({"prompt": prompt, "prediction": response})
    count += 1
    # break
pd.DataFrame(result).to_csv("/scratch/s4702328/cg_backup/test_output4.csv", index=False)
