from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "livmonne/fine-tune_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_response(prompt, max_length=2000):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example test prompt
prompt = "Give me a reading comprehension exercise for a 1st grade student learning English. It should have the word 'red' in it."
response = generate_response(prompt)
print(response)